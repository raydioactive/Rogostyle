library(shiny)
library(plotly)
library(reticulate)
library(uwot) # Make sure this library is installed for UMAP
# Other libraries your script requires

# If you haven't already initialized Python in this R session
use_condaenv("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert", required = TRUE)
use_python("/slipstream_old/home/juliusherzog/miniconda3/envs/latinbert/bin/python")

# Define UI
ui <- fluidPage(
  # Full-width Plotly output, height adjusts to window size
  plotlyOutput("plotly_umap", width = "100%", height = "800px")
)

