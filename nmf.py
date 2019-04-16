
from Sig_GPU import *
#from iSig_GPU import *

class NMF(object):

    def __init__(self, X, K0=None, objective='poisson', max_iter=10000, del_=1, tolerance=1e-6, phi=1.0, a=10.0, b=None, prior_on_W='L1', prior_on_H='L1', report_frequency=100, parameters=None, dtype='Float32', active_thresh=1e-5, num_processes='auto'):
        """
        Parse input to run NMF.

        """
        self.X = X
        self.objective = objective
        self.parameters = parameters
        self.mu = np.mean(self.X.values)
        self.var = np.var(self.X.values)

        if dtype == 'Float32':
            self.dtype = torch.float32
        elif dtype == 'Float16':
            self.dtype = torch.float16

        if self.objective == 'poisson':
            self.Beta = 1
        elif self.objective == 'gaussian':
            self.Beta = 2
        else:
            ValueError("Objective should be either 'gaussian' or 'poisson'")

        start_alg_time = time.time()

        data = ARD_NMF(self.X, self.objective)

        self.channel_names = data.channel_names
        self.sample_names = data.sample_names

        if phi == -1:
            print("Computing dispersion parameter (phi) automatically.")
            self.phi = self.compute_phi()
        else:
            self.phi = phi

        self.W, self.H, self.cost = run_method_engine(data, a, self.phi, b, self.Beta, prior_on_W, prior_on_H, K0, tolerance, max_iter)
        self.alg_time = time.time() - start_alg_time

        self.W, self.H, self.nsig = self.normalize_nmf_output()
        self.W, self.H = self.nmf_to_pd()

    def compute_nmf_result(self, cut_norm=0.5, cut_diff=1.0):
        """Assigns NMF_Result object to result attribute."""
        self.result = NMF_Result(self.X, self.W, self.H, cut_norm=cut_norm, cut_diff=cut_diff, objective=self.objective)

    def compute_phi(self):
        """
        Compute dispersion parameter (phi).
        """
        return self.var / (self.mu ** (2 - self.Beta))

    def normalize_nmf_output(self, active_thresh=1e-5):
        """
        Prunes output from ARD-NMF.
        """
        nonzero_idx = (np.sum(self.H, axis=1) * np.sum(self.W, axis=0)) > active_thresh
        W_active = self.W[:, nonzero_idx]
        H_active = self.H[nonzero_idx, :]
        nsig = np.sum(nonzero_idx)

        # Normalize W and transfer weight to H matrix
        W_weight = np.sum(W_active, axis=0)
        W_final = W_active / W_weight
        H_final = W_weight[:, np.newaxis] * H_active

        return W_final, H_final, nsig

    def nmf_to_pd(self):
        """
        Collapse NMF output to pandas dataframe.
        """
        sig_names = [str(i) for i in range(1,self.nsig+1)]
        return pd.DataFrame(data=self.W, index=self.channel_names, columns=sig_names), pd.DataFrame(data=self.H, index=sig_names, columns=self.sample_names)

# ---------------------------------
# NMF Helper Functions
# ---------------------------------
def scale_norm_nmf(W, H):
    """
    Input is pandas dataframes.
    Scales and normalizes W and X matrices resulting from NMF.

    Code adapted from Jaegil Kim.
    """
    if isinstance(W,str):
        try:
            W=pd.read_csv(W, sep='\t', index_col=0)
            W.columns = [w[1:] if w[0]=='W' else w for w in list(W)]
        except:
            raise ValueError("Please provide valid W input file.")

    if isinstance(H,str):
        try:
            H=pd.read_csv(H, sep='\t', index_col=0)
            H.index = [h[1:] if h[0]=='W' else h for h in H.index]
        except:
            raise ValueError("Please provide valid H input file.")

    Wnorm = W.copy()
    Hnorm = H.copy()

    # Scale Matrix
    for j in range(W.shape[1]):
        Wnorm.iloc[:,j] *= H.sum(1).values[j]
        Hnorm.iloc[j,:] *= W.sum(0).values[j]

    # Normalize
    Wnorm = Wnorm.div(Wnorm.sum(1),axis=0)
    Hnorm = Hnorm.div(Hnorm.sum(0),axis=1)

    H = H.T
    Hnorm = Hnorm.T

    # Get Max Values
    H['max_id']=H.idxmax(axis=1, skipna=True).astype('int')
    H['max']=H.max(axis=1, skipna=True)
    Hnorm['max_norm']=Hnorm.max(axis=1, skipna=True)

    W['max_id']=W.idxmax(axis=1, skipna=True).astype('int')
    W['max']=W.max(axis=1, skipna=True)
    Wnorm['max_norm']=Wnorm.max(axis=1, skipna=True)

    H['max_norm'] = Hnorm['max_norm']
    W['max_norm'] = Wnorm['max_norm']

    return W,H

def select_markers_nmf(X, W, H, cut_norm=0.5, cut_diff=1.0):
    """
    Select from bayesian NMF run.
    Returns:
        - Marker matrix
        - Cells --> Cluster ID

    Code adapted from Jaegil Kim.
    """
    markers = list()
    pd.options.mode.chained_assignment = None
    for n in tqdm(np.unique(W['max_id']), desc='Clusters: '):
        if H[H['max_id']==n].shape[0] > 0:
            tmp = W[W['max_id']==n]
            tmp.loc[:,'mean_on'] = X.loc[np.array(tmp.index), H[H['max_id']==n].index].mean(axis=1)
            tmp.loc[:,'mean_off'] = X.loc[np.array(tmp.index), H[H['max_id']!=n].index].mean(axis=1)
            tmp.loc[:,'diff'] = tmp.loc[:,'mean_on'] - tmp.loc[:,'mean_off']

            tmp.sort_values('diff', ascending=False, inplace=True)
            markers.append(tmp[(tmp['diff'] > cut_diff) & (tmp['max_norm'] >= cut_norm)])

    cell_clusters = H.max_id.sort_values().to_frame()
    nmf_markers = X.loc[pd.concat(markers).index,cell_clusters.index]
    nmf_markers.index.name = 'gene'

    return nmf_markers, cell_clusters

class NMF_Result(object):

    def __init__(self, X, W, H, cut_norm=0.5, cut_diff=1.0, objective='gaussian'):
        """
        NMF_Result class processes and stores information about the run from bayes-NMF.
        Helpful plotting functions provided as well.
        """
        self.objective=objective
        self.W, self.H = scale_norm_nmf(W, H) # Adds processed max_id columns
        self.X_marker, self.g_bayes = select_markers_nmf(X, self.W, self.H, cut_norm=cut_norm, cut_diff=cut_diff) # Computes top markers
        self.tsne_df = None

    def compute_TSNE(self, n_components=2, n_jobs=None, perplexity=30, learning_rate=200.0, n_iter=1000):
        """Compute TSNE"""

        if n_jobs is not None:
            self.tsne_df = pd.DataFrame(mTSNE(verbose=1, n_components=n_components, n_jobs=n_jobs, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(self.H.drop(columns=['max_id','max','max_norm']).values))
        else:
            self.tsne_df = pd.DataFrame(TSNE(n_components=n_components).fit_transform(self.H.drop(columns=['max_id','max','max_norm']).values))

        self.tsne_df['class'] = self.H['max_id'].values

    def compute_alignments(self, metadata_df, batch='batch'):
        """
        Takes in metadata from scanpy object.

        Joins the metadata from original preprocessing to tSNE of H matrix.
        We then compute correlations between joined samples to compute a metric of alignment.
        """

        # Aggregate metadata
        metadata_vars = list(metadata_df) + ['max_id']
        x = self.H.join(metadata_df).loc[:,metadata_vars]

        # if tSNE is computed, add
        if self.tsne_df is not None:
            self.tsne_df.index = x.index
            self.tsne_df = self.tsne_df.join(x)

        # Compute correlation between samples and batches
        corr = m_pearsonr(one_hot(x[batch].values), self.H.drop(['max_id','max','max_norm'],axis=1).values)
        self.corr_df = pd.DataFrame(data = corr, columns=list(self.H)[:-3])

        # Compute single cell counts for each batch/assigned class
        self.count_df = x.groupby([batch,'max_id']).count().iloc[:,[0]].unstack('max_id')
        self.count_df.columns = [str(x) for x in sorted(list(set(x['max_id'])))]
        self.count_df = self.count_df.loc[:,list(self.corr_df)].fillna(int(0)).astype('int')
        sc_counts = self.count_df.values

        # Scale correlations
        scaled_corr = corr * (sc_counts / (np.sum(sc_counts,axis=1)[:,np.newaxis]))
        self.scaled_corr_df = pd.DataFrame(data=scaled_corr, columns=list(self.count_df), index=self.count_df.index)

    def plot_TSNE(self, figsize=(10,10), alpha=0.3, s=5, save_fig=None, by='class'):
        """Plot tSNE"""
        if self.tsne_df is None:
            self.compute_TSNE()

        fig, ax = plt.subplots(figsize=figsize)
        sns.set_style('white')
        sns.scatterplot(x=0, y=1, hue=by, data=self.tsne_df, ax=ax, alpha=alpha, edgecolor=None, s=s, rasterized=True, palette=get_color_cycle(len(set(self.tsne_df['class']))))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('tSNE1')
        plt.ylabel('tSNE2')

        if save_fig is not None:
            plt.savefig(save_fig, format='pdf', dpi=300)

    def plot_TSNE_set(self, figsize=(10,10), save_fig=None):
        """Plot set of curated TSNEs."""
        sns.set_style('white')
        fig, axes = plt.subplots(3,1, figsize=figsize)

        sns.scatterplot(ax=axes[0],x=0,y=1, hue='n_genes', data=self.tsne_df, edgecolor=None, alpha = 0.7, s=15)
        annotate_cluster_names(axes[0],self.tsne_df)
        axes[0].set_xlabel('tSNE1')
        axes[0].set_ylabel('tSNE2')

        sns.scatterplot(ax=axes[1],x=0,y=1, hue='batch', data=self.tsne_df, edgecolor=None, alpha = 0.1, s=10, palette=get_color_cycle(len(set(self.tsne_df['batch']))))
        draw_pie_labels(axes[1], self.tsne_df, alpha=0.8, lw=8, size=500, offset=2.2)
        axes[1].set_xlabel('tSNE1')
        axes[1].set_ylabel('tSNE2')

        sns.scatterplot(ax=axes[2],x=0,y=1, hue='class', data=self.tsne_df, edgecolor=None, alpha = 0.2, s=15, palette=get_color_cycle(len(set(self.tsne_df['class']))))
        axes[2].get_legend().remove()
        annotate_cluster_names(axes[2],self.tsne_df)
        axes[2].set_xlabel('tSNE1')
        axes[2].set_ylabel('tSNE2')

        if save_fig is not None:
            plt.savefig(save_fig, format='pdf', dpi=300)

    def plot_marker_map(self, figsize=(30,15), save_fig=None):
        """Plot marker map."""
        fig, ax = plt.subplots(figsize=figsize)

        if self.objective == 'poisson':
            sns.heatmap(np.log1p(self.X_marker), ax=ax, cmap="YlGnBu", rasterized=True)
        else:
            sns.heatmap(self.X_marker, ax=ax, cmap="YlGnBu", rasterized=True)

        v,c = np.unique(self.g_bayes['max_id'],return_counts=True)
        ax.vlines(np.cumsum(c), *ax.get_ylim())
        ax.set_xticks(np.cumsum(c)-c/2)
        ax.set_xticklabels(v, rotation=360,fontsize=14)

        ax.set_yticks(np.arange(self.X_marker.index.values.shape[0]))
        ax.set_yticklabels(self.X_marker.index.values, fontsize=5)

        if save_fig is not None:
            plt.savefig(save_fig, format='pdf', dpi=300)
