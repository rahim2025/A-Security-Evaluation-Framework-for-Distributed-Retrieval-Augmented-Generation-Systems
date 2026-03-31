import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional
import numpy as np

class Plotter:
    """
    An enhanced class to generate various plots for research with consistent styling
    and improved legend handling.
    """

    def __init__(
        self,
        style: str = "ticks",
        font: str = "Times New Roman",
        height: float = 4,
        aspect: float = 3/2,
        font_scale: float = 1.2,
        num_rows: int = 1,
        num_cols: int = 1,
        legend_spacing: float = 0.06,
        subplot_title_spacing: float = 0.35,
    ):
        """
        Initialize the Plotter with enhanced style settings.

        Args:
            style: Seaborn style (e.g., "ticks", "darkgrid")
            font: Font family for the plot
            height: Height of each subplot in inches
            aspect: Aspect ratio of each subplot (width/height)
            num_rows: Number of rows for subplots
            num_cols: Number of columns for subplots in a row
            legend_spacing: Spacing between legend and plots (0-1)
        """
        self.style = style
        self.font = font
        self.height = height
        self.aspect = aspect
        self.font_scale = font_scale
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.legend_spacing = legend_spacing
        self.subplot_title_spacing = subplot_title_spacing
        self.fig = None
        self.axes = None
        self.legend = None
        self.legend_cols = 1
        self._setup_style()
        self._create_figure()

        self.current_ax_idx = 0

    def _setup_style(self) -> None:
        """Set up the initial plotting style with enhanced defaults."""
        sns.set_theme(style=self.style, font=self.font, font_scale=self.font_scale)

    def _create_figure(self) -> None:
        """Create figure and axes with improved spacing."""
        total_height = self.height * self.num_rows
        total_width = self.height * self.aspect * self.num_cols
        
        self.fig, self.axes = plt.subplots(
            self.num_rows, 
            self.num_cols,
            figsize=(total_width, total_height)
        )

        # Handle single subplot case
        if self.num_rows * self.num_cols == 1:
            self.axes = np.array([self.axes])
        else:
            self.axes = np.array(self.axes).flatten()

        # Add proper spacing between subplots
        self.fig.tight_layout(
            # pad=1.2,
            rect=(0, 0, 1, 1),
        )

    def _load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load and preprocess data with error handling.

        Args:
            data_path: Path to the data file
            **kwargs: Additional arguments for data processing
                     melted (bool): Whether to melt the dataframe
                     id_vars (list): Columns to use as identifier variables
                     var_name (str): Name for the variable column
                     value_name (str): Name for the value column

        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(data_path)
            
            if kwargs.get("melted", False):
                df = df.melt(
                    id_vars=kwargs.get("id_vars"),
                    var_name=kwargs.get("var_name"),
                    value_name=kwargs.get("value_name")
                )
            return df
        except Exception as e:
            raise ValueError(f"Error loading data from {data_path}: {str(e)}")

    def _create_line_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        ax: plt.Axes,
        **kwargs
    ) -> None:
        """Create a line plot with enhanced features."""
        sns.lineplot(data=df, x=x, y=y, ax=ax, **kwargs)
        
        if kwargs.get("annotate"):
            self._add_annotations(df, x, y, ax, kwargs["annotate"])
            
        if kwargs.get("error_bars"):
            self._add_error_bars(df, x, y, ax, kwargs["error_bars"])

    def _create_bar_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        ax: plt.Axes,
        **kwargs
    ) -> None:
        """Create a bar plot with enhanced features."""
        sns.barplot(data=df, x=x, y=y, ax=ax, **kwargs)

    def _create_cat_plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        **kwargs
    ) -> sns.FacetGrid:
        """Create a categorical plot with enhanced features."""
        kind = kwargs.pop("kind", "bar")
        grid = sns.catplot(
            data=df,
            kind=kind,
            x=x,
            y=y,
            height=self.height,
            aspect=self.aspect,
            **kwargs
        )
        
        if kwargs.get("ylim"):
            grid.set(ylim=kwargs["ylim"])
            
        if kwargs.get("hue"):
            self._adjust_catplot_legend(grid, **kwargs)
            
        return grid

    def _customize_axis(self, ax: plt.Axes, **kwargs) -> None:
        """Apply axis customizations."""
        if kwargs.get("ylim"):
            ax.set(ylim=kwargs["ylim"])
            
        if ax.get_legend():
            ax.get_legend().remove()

    def _add_annotations(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        ax: plt.Axes,
        annotate_col: str
    ) -> None:
        """Add annotations to the plot."""
        for _, row in df.iterrows():
            ax.text(
                row[x],
                row[y],
                str(row[annotate_col]),
                fontsize=9.5,
                ha='center',
                va='bottom'
            )

    def _add_error_bars(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        ax: plt.Axes,
        error_col: str
    ) -> None:
        """Add error bars to the plot."""
        ax.errorbar(
            df[x],
            df[y],
            yerr=df[error_col],
            fmt='none',
            capsize=5,
            color='gray',
            alpha=0.5
        )

    def _adjust_catplot_legend(
        self,
        grid: sns.FacetGrid,
        **kwargs
    ) -> None:
        """Adjust catplot legend position with proper spacing."""
        sns.move_legend(
            grid,
            "lower center",
            bbox_to_anchor=(0.5, 1 + self.legend_spacing),
            ncol=self.legend_cols
        )
        grid.figure.tight_layout(
            pad=0.2,
            rect=(0, 0, 1, 0.9)
        )

    def _set_subplot_title(
        self,
        ax: plt.Axes,
        title: str,
        **kwargs
    ) -> None:
        """
        Set subplot title with enhanced formatting options.

        Args:
            ax: The axes object for the subplot
            title: The title text
            **kwargs: Additional formatting parameters
        """
        ax.text(
            0.5,
            -self.subplot_title_spacing,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    def plot(
        self,
        data_path: str,
        plot_type: str,
        x: str,
        y: str,
        subplot_title: str=None,
        show_grid: bool=False,
        **kwargs
    ) -> Optional[sns.FacetGrid]:
        """
        Generate a plot with enhanced features and error handling.

        Args:
            data_path: Path to the CSV data file
            plot_type: Type of plot ("line", "bar", or "catplot")
            x: Column name for x-axis
            y: Column name for y-axis
            subplot_title: Subplot title
            **kwargs: Additional plotting parameters
                     hue (str): Column name for color encoding
                     style (str): Column name for style encoding
                     markers (bool/list): Marker settings
                     palette (str): Color palette
                     ylim (tuple): y-axis limits
                     annotate (str): Column name for annotations
                     error_bars (str): Column name for error bars
                     kind (str): Kind of catplot
        """
        df = self._load_data(data_path, **kwargs)
        ax = self.axes[self.current_ax_idx]
        
        try:
            if plot_type == "line":
                self._create_line_plot(df, x, y, ax, **kwargs)
            elif plot_type == "bar":
                self._create_bar_plot(df, x, y, ax, **kwargs)
            elif plot_type == "catplot":
                return self._create_cat_plot(df, x, y, **kwargs)
            else:
                raise ValueError(f"Invalid plot_type: {plot_type}")

            self._customize_axis(ax, **kwargs)

            if subplot_title:
                self._set_subplot_title(ax, subplot_title)

            if show_grid:
                ax.grid()
            
            self.current_ax_idx += 1
            
        except Exception as e:
            raise ValueError(f"Error creating {plot_type} plot: {str(e)}")

    def add_legend(self, **kwargs) -> None:
        """
        Add a unified legend with improved positioning.

        Args:
            **kwargs: Additional arguments
                     legend_cols (int): Number of columns in the legend
        """
        self.legend_cols = kwargs.get("legend_cols", self.legend_cols)
        handles, labels = self.axes[0].get_legend_handles_labels()

        if self.legend is None:
            self.legend = (handles, labels)
        else:
            existing_handles, existing_labels = self.legend
            self.legend = (
                existing_handles + handles,
                existing_labels + labels
            )

    def save_or_show(self, save_path: str = "") -> None:
        """
        Save or display the plot with improved legend positioning.

        Args:
            save_path: Path to save the plot (if empty, display the plot)
        """
        legend_height = 0
        if self.legend:
            handles, labels = self.legend
            by_label = dict(zip(labels, handles))
            
            # Calculate optimal legend layout
            num_entries = len(by_label)
            legend_rows = (num_entries + self.legend_cols - 1) // self.legend_cols
            legend_height = legend_rows * 0.05
            
            # Adjust subplot positions to make room for legend
            self.fig.subplots_adjust(
                top=1 - legend_height - self.legend_spacing
            )
            
            # Add legend with improved positioning
            self.fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='upper center',
                bbox_to_anchor=(0.5, 1 - (legend_height / 4)),
                ncol=self.legend_cols,
                frameon=True,
                fancybox=True,
                # shadow=True
            )

        # Final layout adjustments
        self.fig.tight_layout(
            pad=0.4,
            rect=(0, 0, 1, 1 - legend_height - self.legend_spacing)
        )

        if save_path:
            plt.savefig(
                save_path,
                format=save_path.split(".")[-1],
                bbox_inches='tight',
                dpi=300
            )
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
