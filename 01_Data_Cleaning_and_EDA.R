# Load necessary pacakges
library(readxl)
library(DataExplorer)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(ROSE)     
library(dplyr)
library(psych)
library(stats)
library(ggfortify)
library(rgl)
library(umap)
library(Rtsne)
library(forcats)
library(car)

# Load and prepare data 
df <- read_excel("C:\\Users\\sophi\\Downloads\\online_shoppers_intention.xlsx")
Online_Shopping <- data.frame(df)

# Rename columns for clarity 
Online_Shopping <- Online_Shopping %>%
  rename(Num_Admin_Pages = Administrative,
         Time_Admin_Pages = Administrative_Duration,
         Num_Info_Pages = Informational,
         Time_Info_Pages = Informational_Duration,
         Num_Product_Pages = ProductRelated,
         Time_Product_Pages = ProductRelated_Duration,
         Bounce_Rate = BounceRates,
         Exit_Rate = ExitRates,
         Page_Value_Score = PageValues,
         Special_Day_Proximity = SpecialDay,
         Operating_System = OperatingSystems,
         User_Region = Region,
         Traffic_Type = TrafficType,
         Visitor_Type = VisitorType,
         Is_Weekend = Weekend,
         Visit_Month = Month,
         Is_Revenue = Revenue)


#Initial checks
str(Online_Shopping)
colSums(is.na(Online_Shopping))
sum(duplicated(Online_Shopping))

# Convert target to a factor
Online_Shopping$Is_Revenue <- as.factor(Online_Shopping$Is_Revenue)
levels(Online_Shopping$Is_Revenue) <- c("No", "Yes")

# Check class imbalance
table(Online_Shopping$Is_Revenue)
prop.table(table(Online_Shopping$Is_Revenue))

# ==============================================================================
# Correlation Analysis Between Numeric Variables
# ==============================================================================

# Convert binary variable to numeric (0/1) for correlation analysis
Online_Shopping$Is_Revenue <- ifelse(Online_Shopping$Is_Revenue == "Yes", 1, 0)

# Select numeric variables for correlation analysis 
numeric_variables <- c("Num_Admin_Pages", "Time_Admin_Pages", 
                       "Num_Info_Pages", "Time_Info_Pages", 
                       "Num_Product_Pages", "Time_Product_Pages", 
                       "Bounce_Rate", "Exit_Rate", 
                       "Page_Value_Score", "Special_Day_Proximity", 
                       "Is_Revenue")

# Compute correlation matrix using pairwise complete observation
correlation_matrix <- cor(Online_Shopping[, numeric_variables], 
                          use = "pairwise.complete.obs")

# Clean labels for visualization 
clean_names <- gsub("_", " ", colnames(correlation_matrix))
colnames(correlation_matrix) <- clean_names
rownames(correlation_matrix) <- clean_names

# Create correlation visualization
corrplot(correlation_matrix, method = "circle", 
         type = "upper", tl.col = "black", tl.srt = 45, 
         title = "Correlation Matrix of Numeric Variables", 
         mar = c(0, 0, 1, 0))

# ==============================================================================
# Review and clean categorical and numeric features
# ==============================================================================

# Convert categorical features to factors 
Online_Shopping$Is_Weekend <- as.factor(Online_Shopping$Is_Weekend)
Online_Shopping$Visit_Month <- as.factor(Online_Shopping$Visit_Month)
Online_Shopping$Visitor_Type <- as.factor(Online_Shopping$Visitor_Type)
Online_Shopping$Operating_System <- as.factor(Online_Shopping$Operating_System)
Online_Shopping$User_Region <- as.factor(Online_Shopping$User_Region)
Online_Shopping$Traffic_Type <- as.factor(Online_Shopping$Traffic_Type)
Online_Shopping$Browser <- as.factor(Online_Shopping$Browser)

# Log transformation
log_vars <- c("Page_Value_Score", "Time_Admin_Pages",
              "Time_Product_Pages", 
              "Time_Info_Pages" )
for (var in log_vars){
  new_var <- paste0("Log_", var)
  Online_Shopping[[new_var]] <- log(Online_Shopping[[var]] + 1)
}

# Discretize some numeric vars
Online_Shopping <- Online_Shopping %>%
  mutate(
    Num_Info_Pages = ntile(Num_Info_Pages, 4), 
    Num_Admin_Pages = ntile(Num_Admin_Pages, 4),
    Num_Product_Pages= ntile(Num_Product_Pages, 4)
  )

# Convert discretized vars to factors 
Online_Shopping$Num_Info_Pages <- factor(Online_Shopping$Num_Info_Pages)
Online_Shopping$Num_Admin_Pages <- factor(Online_Shopping$Num_Admin_Pages)
Online_Shopping$Num_Product_Pages <- factor(Online_Shopping$Num_Product_Pages)

# Check number of levels in each categorical variable 
# Observation:'Traffic_Type' (20 levels) and 'Browser' (13 levels) 
sapply(Online_Shopping[sapply(Online_Shopping, is.factor)], nlevels)

# Group infrequent levels in 'Traffic_Type' and 'Browser' (< 5% occurrence)
Online_Shopping$Traffic_Type <- fct_lump(Online_Shopping$Traffic_Type, 
                                         prop = 0.05)
Online_Shopping$Browser <- fct_lump(Online_Shopping$Browser, 
                                    prop = 0.05)
# Ensure updated variables are treated as factors  
Online_Shopping$Traffic_Type <- factor(Online_Shopping$Traffic_Type)
Online_Shopping$Browser <- factor(Online_Shopping$Browser)

# Remove untransformed highly skewed features from the dataset
Online_Shopping <- Online_Shopping %>% select (-Page_Value_Score,
                                               -Time_Admin_Pages,
                                               -Time_Product_Pages, 
                                               -Time_Info_Pages)

#===============================================================================
# Check Variable Inflation Factor for Numeric Features 
#===============================================================================

# Check multicollinearity using VIF
logit_model <- glm(Is_Revenue ~ ., data = Online_Shopping, family = "binomial")
vif_values <- vif(logit_model)

# Print results
vif_values

# ==============================================================================
# Visualization of PCA Results
# ==============================================================================
# Set seed for reproducibility
set.seed(42)

# Create a string vector for numeric columns to use in PCA
numeric_features <- c("Num_Admin_Pages", "Time_Admin_Pages", 
                      "Num_Info_Pages", "Time_Info_Pages", 
                      "Num_Product_Pages", "Time_Product_Pages", 
                      "Bounce_Rate", "Exit_Rate", 
                      "Page_Value_Score", "Special_Day_Proximity")

# Make sure numeric_features are numeric
numeric_features <- Online_Shopping %>% select_if(is.numeric)

# Check dimensions of the numeric data
print(dim(numeric_features))

# Scale the numeric data
numeric_scaled <- scale(numeric_features)

# Run PCA
pca <- prcomp(numeric_scaled)

# Check PCA results
summary(pca)

# Ensure Is_Revenue is a factor for coloring 
Online_Shopping$Is_Revenue <- as.factor(Online_Shopping$Is_Revenue)

# Rename levels of Is_Revenue to "Yes" and "No" for plotting
levels(Online_Shopping$Is_Revenue) <- c("No", "Yes")

# Visualize PCA results
autoplot(pca, data = Online_Shopping, colour = 'Is_Revenue', 
         loadings.colour = "black", loadings.label.colour = "black") +
  labs(title = "Principal Component Analysis of Numeric Features",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_classic() +
  scale_color_manual(values = c("turquoise", "salmon"), name = "Revenue") +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

# ==============================================================================
# Visualization of PCA Results
# ==============================================================================
# Set seed for reproducibility 
set.seed(42)

# Create a plot showing the proportion of variance explained by PC(1-3)
pca_scores <- as.data.frame(pca$x)
pca_scores$label <- as.factor(Online_Shopping$Is_Revenue)
labels <- pca_scores$label

# Define clean colors 
colors <- c("blue", "#ff7f0e") 
point_colors <- colors[as.numeric(labels)]

# Set background and materials 
bg3d(color = "white")
rgl.clear()

# Create 3D scatterplot 
plot3d(
  x= pca_scores$PC1,
  y = pca_scores$PC2, 
  z = pca_scores$PC3, 
  col = point_colors, 
  size = 3,
  alpha = 0.6,
  type = "s",
  xlab = "PC1", ylab = "PC2", zlab = "PC3"
)

# Add lighting and grid 
rgl.viewpoint(view3d)
grid3d(c("x", "y", "z"), col = "grey80")
legend3d("topright", legend = levels(labels), col = colors, pch = 16, cex = 1.2)

# Save as image (static)
rgl.snapshot("my_3d_plot.png")

#===============================================================================
# t-SNE Visualization of Features
# ==============================================================================

# Set seed for reproducibility
set.seed(42)

# Define numeric and categorical features 
numeric_features <- c("Num_Admin_Pages", "Log_Time_Admin_Pages", 
                      "Num_Info_Pages", "Log_Time_Info_Pages", 
                      "Num_Product_Pages", "Log_Time_Product_Pages", 
                      "Bounce_Rate", "Exit_Rate", "Special_Day_Proximity",
                      "Log_Page_Value_Score")

categorical_features <- c("Visit_Month", "Operating_System", "Browser", 
                          "User_Region", "Visitor_Type", "Is_Weekend", 
                          "Traffic_Type")

# Create a model matrix (one-hot encoding categorical + numeric features)
data_tsne <- model.matrix(~ . - 1, data = Online_Shopping[, c(numeric_features, 
                                                         categorical_features)])
data_tsne <- as.matrix(data_tsne)
is.numeric(data_tsne)

# Add tiny jitter to numeric data to reduce exact duplicates
set.seed(42)
noise_level <- 1e-6
jitter_matrix <- matrix(rnorm(n = length(data_tsne), 
                              mean = 0, sd = noise_level), 
                        nrow = nrow(data_tsne),
                        ncol = ncol(data_tsne))
data_tsne_jittered <- data_tsne + jitter_matrix

# Scale and run tsne 
tsne_out <- Rtsne(scale(data_tsne_jittered), perplexity = 5)

# Prepare results for plotting
df_tsne <- data.frame(tsne_out$Y) 
df_tsne$Revenue <- Online_Shopping$Is_Revenue

# Plot t-SNE results colored by Revenue
ggplot(df_tsne, aes(x = X1, y = X2, color = Revenue)) +
  geom_point(alpha = 0.5) +
  labs(title = "t-SNE Visualisation",
       x = "Dimension 1", y = "Dimension 2") +
  theme_classic() +
  scale_color_manual(values = c("blue", "red"), name = "Revenue")+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# UMAP Visualization on Numeric Features
#===============================================================================

# Set seed for reproducibility
set.seed(42)  

# Extract numeric features and convert to matrix
numeric_features <- Online_Shopping %>% select_if(is.numeric)
features_matrix <- as.matrix(numeric_features)

#Run UMAP
umap_result <- umap(features_matrix)

# Prepare data frame for visualization
umap_df <- as.data.frame(umap_result$layout)
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df$Is_Revenue<- as.factor(Online_Shopping$Is_Revenue)

# Plot UMAP results colored by Revenue
ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Is_Revenue)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("steelblue", "tomato")) +
  theme_minimal() +
  labs(title = "UMAP Visualisation", color = "Is_Revenue")+
  theme(axis.title.y = element_text(angle = 0, vjust = 0.6))

#===============================================================================
# Balance dataset using ROSE
#===============================================================================

# Set seed for reproducibility 
set.seed(42)
balanced_data <- ROSE(Is_Revenue ~ ., data = Online_Shopping, seed = 1)$data

# Confirm class distribution after balancing
table(balanced_data$Is_Revenue)
prop.table(table(balanced_data$Is_Revenue))

#===============================================================================
# Explore and Scale Numeric Features 
#===============================================================================

# Summary statistics before scaling
numeric_cols<- balanced_data[sapply(balanced_data, is.numeric)]
summary(numeric_cols)
describe(balanced_data[sapply(balanced_data, is.numeric)])

# standardize numeric features to mean=0 and sd=1
numeric_cols <- names(balanced_data)[sapply(balanced_data, is.numeric)]
balanced_data[numeric_cols] <- scale(balanced_data[numeric_cols])

# Verify scaling
summary(balanced_data[numeric_cols])
describe(balanced_data[sapply(balanced_data, is.numeric)])

# Save data for training models

file_path <- file.path("C:", "Users", "sophi", 
                       "OneDrive - University of Surrey", 
                       "Documents", 
                       "Machine Learning Coursework Dataset", 
                       "balanced_data.rds")

saveRDS(balanced_data, file_path)
