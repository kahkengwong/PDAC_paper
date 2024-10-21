library(edgeR)
library(readxl)
library(writexl)
library(tidyverse)

# Load and pre-process the expression matrix
control_data <- read_excel("(Input_file_for_DEGs)_Pancreatic-ca_stage_1-2_n=68_Asymp-ctrl.xlsx")
PDAC_data <- read_excel("(Input_file_for_DEGs)_Pancreatic-ca_stage_1-2_n=68.xlsx")
control_data <- as.data.frame(control_data)
PDAC_data <- as.data.frame(PDAC_data)

# Handling the ID column, combine all datasets, and save the gene names
rownames(control_data) <- control_data[,1]
control_data <- control_data[,-1]
rownames(PDAC_data) <- PDAC_data[,1]
PDAC_data <- PDAC_data[,-1]
all_data <- rbind(control_data, PDAC_data)
gene_names <- colnames(all_data)

# Label samples: 1 for control, 2 for PDAC, and transpose the data
group <- factor(c(rep(1, nrow(control_data)), rep(2, nrow(PDAC_data))))
all_data <- t(all_data)
rownames(all_data) <- gene_names

# Create a DGEList object, filter out low expressed genes, perform normalization, common and tagwise dispersions
dge <- DGEList(counts = all_data, group = group)
keep <- rowSums(cpm(dge) > 1) >= 5
dge <- dge[keep, , keep.lib.sizes=FALSE]
dge <- calcNormFactors(dge)
dge <- estimateCommonDisp(dge)
dge <- estimateTagwiseDisp(dge)

# Perform exact test, and obtain DEGs and their statistics
et <- exactTest(dge)
degs <- topTags(et, n=Inf)
degs$table$FDR <- p.adjust(degs$table$PValue, method = "BH")
degs$table$linearFC <- 2^degs$table$logFC
selected <- degs$table[degs$table$FDR < 0.05,]
selected$Gene <- rownames(selected)

# Log2 transformation and data scaling
all_data_transformed <- log2(all_data + 1)
z_score <- function(x) (x - mean(x)) / sd(x)
all_data_scaled <- apply(all_data_transformed, 2, z_score)

# Convert the matrix to a data frame and export the results
all_data_scaled_df <- as.data.frame(all_data_scaled)
all_data_scaled_df <- cbind(Gene = rownames(all_data_scaled_df), all_data_scaled_df)
write_xlsx(all_data_scaled_df, "PDAC_Asymp-ctrl_scaled.xlsx")
write_xlsx(selected, "PDAC_DEGs.xlsx")
