class PCA_tool:
    def __init__(self, df_x: 'pd.DataFrame', df_y: 'pd.DataFrame' = None):
        self.X: pd.DataFrame = df_x
        self.samples: np.ndarray = self.X.index.values
        self.variables: np.ndarray = self.X.columns.values

        if df_y is not None:
            self.y: pd.DataFrame = df_y

            if self.y.iloc[:, 0].nunique() > 5:
                self.y_scale: str = 'continuous'
            else:
                self.y_scale: str = 'discrete'
        else:
            self.y: pd.DataFrame = None

    def autoscale(self) -> 'pd.DataFrame':
        """
        Standardize the dataset using z-score normalization.

        Returns:
            pd.DataFrame: A DataFrame with scaled values, where each feature has a mean of 0 and a standard deviation of 1.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return pd.DataFrame(
            scaler.fit_transform(self.X), 
            columns=self.variables, 
            index=self.samples
        )

    def perform_pca(self, scale: bool = True) -> str:
        """
        Perform Principal Component Analysis (PCA) on the dataset.

        Args:
            scale (bool): Whether to standardize the data before performing PCA. Defaults to True.

        Returns:
            str: Confirmation message indicating the PCA computation is finished.
        """
        import pandas as pd
        from sklearn.decomposition import PCA

        if scale:
            autoscaled_X = self.autoscale()
            self.model: PCA = PCA()
            self.model.fit(autoscaled_X)
            x_pca: np.ndarray = self.model.transform(autoscaled_X)

            scores_names = [f"PC{i+1}" for i in range(len(self.model.components_))]
            self.scores: pd.DataFrame = pd.DataFrame(x_pca, index=self.samples, columns=scores_names)
            self.eigenvalues: pd.DataFrame = pd.DataFrame(
                self.model.explained_variance_.round(4), 
                index=scores_names, 
                columns=['Eigenvalue']
            )
            self.variance: pd.DataFrame = pd.DataFrame(
                self.model.explained_variance_ratio_.round(4) * 100, 
                index=scores_names, 
                columns=['Percent of variance explained']
            )

            Z_with_scores = autoscaled_X.join(self.scores)
            Z_with_scores_corr = Z_with_scores.corr()
            self.loadings: pd.DataFrame = (
                Z_with_scores_corr.iloc[len(self.variables):, :len(self.variables)]
            ).T
        else:
            self.model: PCA = PCA()
            self.model.fit(self.X)
            x_pca: np.ndarray = self.model.transform(self.X)

            scores_names = [f"PC{i+1}" for i in range(len(self.model.components_))]
            self.scores: pd.DataFrame = pd.DataFrame(x_pca, index=self.samples, columns=scores_names)
            self.eigenvalues: pd.DataFrame = pd.DataFrame(
                self.model.explained_variance_.round(4), 
                index=scores_names, 
                columns=['Eigenvalue']
            )
            self.variance: pd.DataFrame = pd.DataFrame(
                self.model.explained_variance_ratio_.round(4) * 100, 
                index=scores_names, 
                columns=['Percent of variance explained']
            )
            self.loadings: pd.DataFrame = pd.DataFrame(
                self.model.components_.T, 
                index=self.variables, 
                columns=scores_names
            )

        return 'PCA computation finished'

    def save_data(self, filename: str = 'pca_data') -> None:
        """
        Save PCA results to an Excel file.

        Args:
            filename (str): Name of the output Excel file (without extension). Defaults to 'pca_data'.

        Saves:
            An Excel file with separate sheets for scores, eigenvalues, explained variance, and loadings.
        """
        import pandas as pd

        with pd.ExcelWriter(f'{filename}.xlsx') as writer:
            self.scores.to_excel(writer, sheet_name='scores')
            self.eigenvalues.to_excel(writer, sheet_name='eigenvalues')
            self.variance.to_excel(writer, sheet_name='explained variance')
            self.loadings.to_excel(writer, sheet_name='loadings')

    def plot_loadings(self, n_pcs: int = 3, thresh: float = 0.7, show_top: bool = False,
                      save: bool = False, filename: str = 'loadings_plot') -> None:
        """
        Visualize the loadings for the principal components as horizontal bar plots.

        Args:
            n_pcs (int): Number of principal components to display. Defaults to 3.
            thresh (float): Threshold for highlighting loadings. Defaults to 0.7.
            show_top (bool): Whether to limit the plot to the top 10 variables. Defaults to False.
            save (bool): Whether to save the plot as an image file. Defaults to False.
            filename (str): Filename for the saved image (if save is True). Defaults to 'loadings_plot'.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not show_top:
            loadings = self.loadings.iloc[:, :n_pcs]
        else:
            loadings = self.loadings.iloc[:10, :n_pcs]

        col = []
        for x in range(len(loadings.columns)):
            for y in loadings.iloc[:, x]:
                if (y >= thresh) or (y <= -thresh):
                    col.append('#35B2F0')
                else:
                    col.append('#C2CCD0')

        col2 = np.asarray(col).reshape(loadings.shape[1], loadings.shape[0])
        y_pos = np.arange(len(loadings.index))
        plt.figure(figsize=(10, 10))

        for x in range(len(loadings.columns)):
            plt.subplot(1, len(loadings.columns), x + 1)
            plt.barh(y_pos, loadings.iloc[:, x], color=col2[x])
            plt.yticks(y_pos, loadings.index.values)
            plt.xticks([-1, -thresh, 0, thresh, 1])
            plt.axvline(-thresh, c='gray', linestyle='--')
            plt.axvline(thresh, c='gray', linestyle='--')
            plt.grid(alpha=0.5)
            plt.title(f'PC{x+1} loadings')

        plt.tight_layout()

        if save:
            plt.savefig(f'{filename}.jpg', dpi=300)
        plt.show()

    def plot_scatter(self, plot_pcs: list[int] = [1, 2], annot: bool = True, annot_pos: list[float] = [0.1, 0.1],
                     msize: int = 200, color_by_y: bool = True, colorscale: str = 'RdYlBu',
                     alpha: float = 1, edgecolors: str = 'k', figsize: tuple[int, int] = (12, 11),
                     save: bool = False) -> None:

        """
        Create a scatter plot of PCA scores for specified principal components.

        Args:
            plot_pcs (list[int]): Principal components to plot (e.g., [1, 2]). Defaults to [1, 2].
            annot (bool): Whether to annotate the points with sample labels. Defaults to True.
            annot_pos (list[float]): Offset for annotations. Defaults to [0.1, 0.1].
            msize (int): Marker size for scatter points. Defaults to 200.
            color_by_y (bool): Whether to color points by the target variable (y). Defaults to True.
            colorscale (str): Colormap for continuous target variables. Defaults to 'RdYlBu'.
            alpha (float): Transparency level for points. Defaults to 1.
            edgecolors (str): Color of point edges. Defaults to 'k'.
            figsize (tuple[int, int]): Size of the figure. Defaults to (12, 11).
            save (bool): Whether to save the plot as an image file. Defaults to False.

        """ 
        import matplotlib.pyplot as plt
        import numpy as np

        dx, dy = annot_pos
        plt.figure(figsize=figsize)

        if self.y is not None and color_by_y:
            if self.y_scale == 'continuous':
                sc = plt.scatter(
                    self.scores.iloc[:, plot_pcs[0] - 1],
                    self.scores.iloc[:, plot_pcs[1] - 1],
                    s=msize,
                    c=self.y.iloc[:, 0].to_list(),
                    edgecolors=edgecolors,
                    linewidths=1,
                    cmap=colorscale,
                    alpha=alpha
                )
                cbar = plt.colorbar(sc)
                cbar.set_label(f"{self.y.columns.values[0]}", fontsize=15)
                cbar.set_alpha(1)
            else:
                import matplotlib.colors as mcolors

                colors = {i: mcolors.to_hex(plt.cm.tab10(i / 5)) for i in range(5)}
                unique_labels = np.unique(self.y.iloc[:, 0])
                filtered_colors = {label: colors[label] for label in unique_labels}

                for label, color in filtered_colors.items():
                    plt.scatter(
                        self.scores.iloc[:, plot_pcs[0] - 1][self.y.iloc[:, 0] == label],
                        self.scores.iloc[:, plot_pcs[1] - 1][self.y.iloc[:, 0] == label],
                        label=f'Class {label}',
                        color=color,
                        s=msize,
                        edgecolors=edgecolors,
                        linewidths=1,
                        alpha=alpha
                    )
                plt.legend(title='Class', loc='best')
        else:
            print("Warning, no y dataframe was loaded to PCA tool! Continuing without colorby.")
            plt.scatter(
                self.scores.iloc[:, plot_pcs[0] - 1],
                self.scores.iloc[:, plot_pcs[1] - 1],
                s=msize,
                edgecolors=edgecolors,
                linewidths=1,
                alpha=alpha
            )

        if annot:
            x = self.scores.iloc[:, plot_pcs[0] - 1]
            y = self.scores.iloc[:, plot_pcs[1] - 1]
            for i, txt in enumerate(self.samples):
                plt.annotate(txt, (x.iloc[i] + dx, y.iloc[i] + dy), fontsize=12)

        plt.xlabel(
            f'PC{plot_pcs[0]}, explained variance: '
            + format((self.model.explained_variance_ratio_[plot_pcs[0] - 1]) * 100, '.2f') + '%',
            fontsize=15
        )
        plt.ylabel(
            f'PC{plot_pcs[1]}, explained variance: '
            + format((self.model.explained_variance_ratio_[plot_pcs[1] - 1]) * 100, '.2f') + '%',
            fontsize=15
        )
        plt.grid(alpha=alpha)
        plt.title(
            f'PC{plot_pcs[0]} vs. PC{plot_pcs[1]}, total explained variance: '
            + str(round((
                self.model.explained_variance_ratio_[plot_pcs[0] - 1]
                + self.model.explained_variance_ratio_[plot_pcs[1] - 1]), 2) * 100) + '%'
        )
        if save:
            plt.savefig(f'PC{plot_pcs[0]}_PC{plot_pcs[1]}.jpg', dpi=300)
        plt.show()

    def plot_scree(self, filename: str = 'scree_plot', num_pcs: str = 'all', save: bool = False) -> None:
        """
        Generate a scree plot to visualize the variance explained by each principal component.

        Args:
            filename (str): Name of the file to save the plot (if save is True). Defaults to 'scree_plot'.
            num_pcs (str): Number of principal components to include ('all' for all components). Defaults to 'all'.
            save (bool): Whether to save the plot as an image file. Defaults to False.

        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 5))
        if num_pcs == 'all':
            x = np.array(self.eigenvalues['Eigenvalue'])
            plt.xticks(range(len(x)), self.eigenvalues.index.values)
        else:
            x = np.array(self.eigenvalues['Eigenvalue'])[:int(num_pcs)]
            plt.xticks(range(int(num_pcs)), self.eigenvalues.index.values[:int(num_pcs)])

        plt.plot(x)
        plt.title('Scree criterion')
        plt.ylabel('Variance')
        plt.xlabel('Principal components')
        plt.tight_layout()

        if save:
            plt.savefig(f'{filename}.jpg', dpi=300)
        plt.show()

    def __str__(self) -> str:
        return (
            f"Dataset samples: {self.X.shape[0]} \n"
            f"Dataset variables: {self.X.shape[1]}"
        )