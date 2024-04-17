library(edgeR)
library(readxl)
library(writexl)
library(tidyverse)

# Load the control and PDAC expression matrix files
control_data <- read_excel("(Input_file_for_DEGs)_Pancreatic-ca_stage_1-2_n=68_Asymp-ctrl.xlsx")
PDAC_data <- read_excel("(Input_file_for_DEGs)_Pancreatic-ca_stage_1-2_n=68.xlsx")

# Convert data to data frames
control_data <- as.data.frame(control_data)
PDAC_data <- as.data.frame(PDAC_data)

# Set row names as the ID column and then remove this column
rownames(control_data) <- control_data[,1]
control_data <- control_data[,-1]
rownames(PDAC_data) <- PDAC_data[,1]
PDAC_data <- PDAC_data[,-1]

# Combine datasets
all_data <- rbind(control_data, PDAC_data)

# Save the gene names before transposition
gene_names <- colnames(all_data)

# Label samples: 1 for control, 2 for PDAC
group <- factor(c(rep(1, nrow(control_data)), rep(2, nrow(PDAC_data))))

# Transpose all_data so that genes are rows and samples are columns
all_data <- t(all_data)

# Assign gene names as row names
rownames(all_data) <- gene_names

# Create a DGEList object, required for edgeR analysis
dge <- DGEList(counts = all_data, group = group)

# Perform normalization
dge <- calcNormFactors(dge)

# Compute common dispersion
dge <- estimateCommonDisp(dge)

# Compute tagwise dispersion
dge <- estimateTagwiseDisp(dge)

# Perform exact test
et <- exactTest(dge)

# Obtain a table of differentially expressed genes
degs <- topTags(et, n=Inf)

# Adjust p-values for multiple testing using the Benjamini-Hochberg method
degs$table$FDR <- p.adjust(degs$table$PValue, method = "BH")

# Convert log2FC to linear scale
degs$table$linearFC <- 2^degs$table$logFC

# Select the differentially expressed genes with a threshold of 0.05
selected <- degs$table[degs$table$FDR < 0.05,]

# Add gene names to the selected genes data frame
selected$Gene <- rownames(selected)

# Add 1 to the expression values and perform log2 transformation
all_data_transformed <- log2(all_data + 1)

# Scale (z-score) the transformed expression values
z_score <- function(x) (x - mean(x)) / sd(x)
all_data_scaled <- apply(all_data_transformed, 2, z_score)

# Convert the matrix to a data frame
all_data_scaled_df <- as.data.frame(all_data_scaled)

# Add gene names as the first column of the data frame
all_data_scaled_df <- cbind(Gene = rownames(all_data_scaled_df), all_data_scaled_df)

# Write the data frame to an xlsx file
write_xlsx(all_data_scaled_df, "PDAC_Asymp-ctrl_scaled.xlsx")
write_xlsx(selected, "PDAC_DEGs.xlsx")