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

# Define server logic
server <- function(input, output) {
  # Your preprocessing and plot-generating code goes here
  # For simplicity, directly using the plot generation from your script:

  # Source your Python functions
  source_python("~/Rogostyle/gen_berts.py")

  # Assuming you've already generated `umap_df` as shown in your script
  umap_df<-readRDS("~/Rogostyle/UMAP_df.rds")

  output$plotly_umap <- renderPlotly({
    # Ensure this is the plot generation code from your script
    plot_ly(umap_df, x = ~UMAP1, y = ~UMAP2, type = 'scatter', mode = 'markers',
            text = ~Sentence, # Display the sentence on hover
            color = ~Genre, # Color by genre
            marker = list(size = 10)) %>%
      layout(title = 'UMAP Plot of LatinBERT Embeddings',
             xaxis = list(title = 'UMAP1'),
             yaxis = list(title = 'UMAP2'),
             hovermode = 'closest', autosize = T)
  })
}

# Run the application
shinyApp(ui = ui, server = server)
