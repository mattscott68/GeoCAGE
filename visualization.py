import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class VisualEmbedding():

    def __init__(self, data, labels, reduction=["pca","t-sne"]):
        self.data = data
        self.reduction_methods = reduction
        self.labels = labels

    def embedding_visualization(self,path):
        feat_cols = [ 'att_'+str(i) for i in range(len(self.data[0])) ]
        df_data = pd.DataFrame(self.data,columns=feat_cols)
        df_data['y'] = self.labels
        df_data['label'] = df_data['y'].apply(lambda i: str(i))

        df_visual = df_data.copy()

        #pca-2d
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df_data[feat_cols].values)
        df_visual['pca-one'] = pca_result[:,0]
        df_visual['pca-two'] = pca_result[:,1]
        df_visual['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        #t-sne-2pca
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(df_data)
        df_visual['tsne-2d-one'] = tsne_results[:,0]
        df_visual['tsne-2d-two'] = tsne_results[:,1]
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


        #t-sne-50pca
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(df_data)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_pca_results = tsne.fit_transform(pca_result_50)
        df_visual['tsne-pca50-one'] = tsne_pca_results[:,0]
        df_visual['tsne-pca50-two'] = tsne_pca_results[:,1]
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        #plot
        plt.figure(figsize=(16,7))
        ax1 = plt.subplot(2, 2, 1)
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_visual,
            legend="full",
            alpha=1
        )

        ax2 = plt.subplot(2, 2, 2)
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
            xs=df_visual["pca-one"],
            ys=df_visual["pca-two"],
            zs=df_visual["pca-three"],
            c=df_visual["y"],
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')

        ax3 = plt.subplot(2, 2, 3)
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_visual,
            legend="full",
            alpha=1
        )

        ax4 = plt.subplot(2, 2, 4)
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-pca50-one", y="tsne-pca50-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df_visual,
            legend="full",
            alpha=1
        )
        plt.show()
        return df_data

