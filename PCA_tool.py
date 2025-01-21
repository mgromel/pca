class PCA_tool:
    def __init__(self, df_x, df_y=None):
        
        self.X = df_x
        self.samples = self.X.index.values
        self.variables = self.X.columns.values

        if df_y is not None:
            self.y = df_y
            
            if self.y.iloc[:,0].nunique() > 5:
                self.y_scale = 'continuous'
            else:
                self.y_scale = 'discrete'
        else:
            self.y = None
        
    def autoscale(self):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(self.X), columns=self.variables, index=self.samples)

    def perform_pca(self, scale=True):
        import pandas as pd
        from sklearn.decomposition import PCA

        # check scaling
        if scale == True:
            autoscaled_X = self.autoscale()
            self.model = PCA()
            # fit and transform
            self.model.fit(autoscaled_X)
            x_pca = self.model.transform(autoscaled_X)
            
            # create PC names
            scores_names = [f"PC{i+1}" for i in range(len(self.model.components_))]

            # create scores, eigenvalues
            self.scores = pd.DataFrame(x_pca, index=self.samples, columns=scores_names)
            self.eigenvalues = pd.DataFrame(self.model.explained_variance_.round(4), index=scores_names, columns=['Eigenvalue'])

            # calculate explained variance
            self.variance=pd.DataFrame(self.model.explained_variance_ratio_.round(4)*100, index=scores_names, columns=['Percent of variance explained'])

            # calculate loadings
            Z_with_scores=autoscaled_X.join(self.scores)
            Z_with_scores_corr=Z_with_scores.corr()
            self.loadings = (Z_with_scores_corr.iloc[len(self.variables):,:len(self.variables)]).T
        else:
            self.model = PCA()
            # fit and transform
            self.model.fit(self.X)
            x_pca = self.model.transform(self.X)
            
            # create PC names
            scores_names = [f"PC{i+1}" for i in range(len(self.model.components_))]

            # create scores, eigenvalues
            self.scores = pd.DataFrame(x_pca, index=self.samples, columns=scores_names)
            self.eigenvalues = pd.DataFrame(self.model.explained_variance_.round(4), index=scores_names, columns=['Eigenvalue'])

            # calculate explained variance
            self.variance=pd.DataFrame(self.model.explained_variance_ratio_.round(4)*100, index=scores_names, columns=['Percent of variance explained'])

            # calculate loadings
            self.loadings = pd.DataFrame(self.model.components_.T, index=self.variables, columns=scores_names)

        return 'PCA computation finished' #self.scores, self.variance, self.eigenvalues, self.loadings

    def save_data(self, filename='pca_data'):
        import pandas as pd
        with pd.ExcelWriter(f'{filename}.xlsx') as writer:
            self.scores.to_excel(writer, sheet_name='scores')
            self.eigenvalues.to_excel(writer, sheet_name='eigenvalues')
            self.variance.to_excel(writer, sheet_name='explained variance')
            self.loadings.to_excel(writer, sheet_name='loadings')

    def plot_loadings(self, n_pcs=3, thresh=0.7, show_top=False, save_fig=False, filename='loadings_plot', save=False):
        import matplotlib.pyplot as plt
        import numpy as np

        if show_top != True:
            loadings = self.loadings.iloc[:,:n_pcs]
        else:
            loadings = self.loadings.iloc[:10,:n_pcs]

        col=[]
        for x in range(0,len(loadings.columns)):
            for y in loadings.iloc[:,x]:
                if (y >= thresh) or (y <= -thresh):
                    col.append('#35B2F0')
                else:
                    col.append('#C2CCD0')
        col2 = np.asarray(col)
        col2 = col2.reshape(loadings.shape[1], loadings.shape[0])

        # vertical barplot for loadings
        y_pos = np.arange(len(loadings.index))
        plt.figure(figsize=(10,10))
        for x in range(0,len(loadings.columns)):
            plt.subplot(1, len(loadings.columns), x+1)
            plt.barh(y_pos, loadings.iloc[:,x], color=col2[x])
            plt.yticks(y_pos, loadings.index.values)
            plt.xticks([-1,-thresh,0,thresh,1])
            plt.axvline(-thresh, c='gray', linestyle='--')
            plt.axvline(thresh, c='gray', linestyle='--')
            plt.grid(alpha=0.5)
            plt.title('PC'+ str(x+1) + ' loadings')
        plt.tight_layout()

        if save == True:
            plt.savefig(f'{filename}.jpg', dpi=300)
        plt.show()

    def plot_scatter( self, 
                      plot_pcs=[1,2],
                      annot=True,
                      annot_pos=[0.1, 0.1], 
                      msize=200,
                      color_by_y=True,
                      colorscale='RdYlBu',
                      
                      alpha = 1,
                      edgecolors = 'k',
                      figsize = (12,11),

                      save = False,
                      ):

        import matplotlib.pyplot as plt
        import numpy as np
        #pozycje etykiet
        dx=annot_pos[0]
        dy=annot_pos[1]


        #size markerów
        msize=msize

        plt.figure(figsize = figsize)

        if ((self.y is not None) and (color_by_y == True)):

            if self.y_scale == 'continuous':
                sc= plt.scatter(self.scores.iloc[:,plot_pcs[0]-1], self.scores.iloc[:,plot_pcs[1]-1],
                                s=msize,
                                c=self.y.iloc[:,0].to_list(),
                                edgecolors=edgecolors,
                                linewidths=1,
                                cmap=colorscale,
                                alpha=alpha)
                # colorbar
                cbar = plt.colorbar(sc)
                cbar.set_label(f"{self.y.columns.values[0]}", fontsize=15)
                cbar.set_alpha(1)
            else:
                import matplotlib.colors as mcolors

                colors = {i: mcolors.to_hex(plt.cm.tab10(i / 5)) for i in range(5)}
                
                unique_labels = np.unique(self.y.iloc[:,0])
                filtered_colors = {label: colors[label] for label in unique_labels}

                
                for label, color in filtered_colors.items():
                    plt.scatter(self.scores.iloc[:,plot_pcs[0]-1][self.y.iloc[:,0] == label], 
                                self.scores.iloc[:,plot_pcs[1]-1][self.y.iloc[:,0] == label], 
                                label=f'Class {label}', 
                                color=color,
                                s=msize,                
                                edgecolors=edgecolors,
                                linewidths=1,
                                alpha=alpha
                                
                                )
                # Add legend
                plt.legend(title='Class', loc='best')
            
        else:
            print("Warning, no y dataframe was loaded to PCA tool! Continuing without colorby.")
            sc= plt.scatter(self.scores.iloc[:,plot_pcs[0]-1], self.scores.iloc[:,plot_pcs[1]-1],
                s=msize,
                edgecolors=edgecolors,
                linewidths=1,
                alpha=alpha)
            


        # wł/wył annotacje
        if annot == True:
            x = self.scores.iloc[:, int(plot_pcs[0]-1)]
            y = self.scores.iloc[:, int(plot_pcs[1]-1)]
            for i, txt in enumerate(self.samples):
                plt.annotate(txt, (x.iloc[i]+dx, y.iloc[i]+dy), fontsize=12)

        # osie
        plt.xlabel(f'PC{plot_pcs[0]}, explained variance: ' + format((self.model.explained_variance_ratio_[plot_pcs[0]-1])*100, '.2f') + '%', fontsize='15')
        plt.ylabel(f'PC{plot_pcs[1]}, explained variance: ' + format((self.model.explained_variance_ratio_[plot_pcs[1]-1])*100, '.2f') + '%', fontsize='15')
        plt.grid(alpha=alpha)



        # title
        plt.title(f'PC{plot_pcs[0]} vs. PC{plot_pcs[1]}, total explained variance: {round((self.model.explained_variance_ratio_[plot_pcs[0]-1]+self.model.explained_variance_ratio_[plot_pcs[1]-1]),2)*100}%')
        if save == True:
            plt.savefig(f'PC{plot_pcs[0]}_PC{plot_pcs[1]}.jpg', dpi=300)
        plt.show()

    def plot_scree(self, filename='scree_plot', num_pcs='all', save=False):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10,5))
        if num_pcs == 'all':
            x = np.array(self.eigenvalues['Eigenvalue'])
            plt.xticks(range(0, len(x)), self.eigenvalues.index.values)
        else:
            x = np.array(self.eigenvalues['Eigenvalue'])[:int(num_pcs)]
            plt.xticks(range(0, int(num_pcs)), self.eigenvalues.index.values[:int(num_pcs)])
        plt.plot(x)
        plt.title('Scree criterion')
        plt.ylabel('Variance')
        plt.xlabel('Principal components')
        plt.tight_layout()
        

        if save == True:
            plt.savefig(f'{filename}.jpg', dpi=300)
        plt.show()

    def __str__(self):
        return f"""Dataset samples: {self.df_x.shape[0]} \nDataset variables: {self.df_x.shape[1]}
                """
