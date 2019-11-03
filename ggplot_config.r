# Preferences and common settings for use with ggplot2 visualization package in R
# @author: C Heiser
# November 2019

require('gplots')
require('plotly')


# save ggplot as .png image
to.png <- function(plt, destination = 'plt.png', w = 8, h = 8, r = 700){
  # plt = ggplot object
  # destination = filepath, name, and extension of output
  # w, h = width, height in inches
  # r = resolution
  ggsave(filename = destination, plot = plt, width = w, height = h, units = 'in', dpi = r)
}


# plot multiple complete plot objects on one image
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL){
  # Multiple plot function
  # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
  # - cols:   Number of columns in layout
  # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
  # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
  # then plot 1 will go in the upper left, 2 will go in the upper right, and
  # 3 will go all the way across the bottom.
  library(grid)
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  # If layout is NULL, then use 'cols' to determine layout
  if(is.null(layout)){
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  if(numPlots==1){
    print(plots[[1]])
  }else{
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    # Make each plot, in the correct location
    for(i in 1:numPlots){
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


# color scale
my_colors <- list("green"="#32d339",
                  "yellow"="#cedc00",
                  "red"="#e1261c",
                  "purple"="#74538F",
                  "teal"="#20cbd4",
                  "blue"="#3f97b5",
                  "lightgray"="#c8c9c7",
                  "darkgray"="#54585a")


# preferred plotting options
# call these by adding them (+) to a ggplot object
plot.opts <- list(
  theme_bw(),
  theme(text = element_text(colour = my_colors$darkgray),
        legend.text = element_text(size=9),
        axis.line = element_line(colour = my_colors$darkgray),
        axis.title = element_text(size=12),
        axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=10),
        plot.title = element_text(size=18))
)

# options for dimension reduction plots (t-SNE, UMAP, PCA)
DR.opts <- list(
  theme_bw(),
  theme(panel.grid = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        axis.title = element_text(size=12),
        panel.border = element_blank(),
        axis.line = element_line(arrow = arrow(type = 'closed', length = unit(0.3, 'cm'), angle = 15)))
)
