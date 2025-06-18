Using Julia, train a random forest to see how it performs on the chr1 callvariants TP and FP variants


```julia
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DecisionTree")
Pkg.add("MLDataUtils")
Pkg.add("StatsBase")
Pkg.add("JLD")


using CSV, DataFrames, DecisionTree, MLDataUtils, Statistics, StatsBase, JLD, Random


# read in the tables
positives = CSV.read("hg001-4.herro.Q30.sam1.3.noSoftClip.noChr22.bam.callvariants.raw.norm.tp.txt", DataFrame, delim='\t')
negatives = CSV.read("hg001-4.herro.Q30.sam1.3.noSoftClip.noChr22.bam.callvariants.raw.norm.fp.txt", DataFrame, delim='\t')


# Combine positives and negatives into X
X = vcat(positives, negatives, cols=:union)


# Create y DataFrame with labels
y_pos = fill("TP", nrow(positives))
y_neg = fill("FP", nrow(negatives))
y = DataFrame(Type = vcat(y_pos, y_neg))






# data transformation
dt = fit(ZScoreTransform, Matrix(X), dims=1)


df = DataFrame(StatsBase.transform(dt, Matrix(X)), :auto)


# Function to check for any NaNs in a column
contains_nan(col) = any(isnan.(col))


# Identify columns that contain NaN values
nan_columns = [name for name in names(df) if contains_nan(df[!, name])]


# Remove columns with any NaN values
clean_df = select(df, Not(nan_columns))


# Display the cleaned DataFrame
clean_df


# combine the TP and values
combined_df = hcat(clean_df,y)


# randomly shuffle with a random seed
shuffle_df = shuffle(Random.seed!(123), combined_df)




# get rid of TP column
shuffle_df_no_TP = shuffle_df[:, Not(21)]


# Assuming you have your X and y DataFrames available


# keep just the TP column
shuffle_df_TP = shuffle_df[:, Not(1:20)]


# Split the data into training and testing sets
(train_indices, test_indices) = splitobs(1:nrow(shuffle_df_no_TP), at = 0.8)


train_mask = [i in train_indices for i in 1:nrow(shuffle_df_TP)]
test_mask = [i in test_indices for i in 1:nrow(shuffle_df_TP)]


train_X = shuffle_df_no_TP[train_mask, :]
train_y = shuffle_df_TP[train_mask, :]
test_X = shuffle_df_no_TP[test_mask, :]
test_y = shuffle_df_TP[test_mask, :]


model = RandomForestClassifier(n_trees=100, max_depth=60, rng=123, min_samples_split=5)#RandomForestClassifier
#n_trees:             100
#n_subfeatures:       -1
#partial_sampling:    0.7
#max_depth:           4
#min_samples_leaf:    1
#min_samples_split:   2
#min_purity_increase: 0.0
#classes:             nothing
#ensemble:            nothing


@time DecisionTree.fit!(model, Matrix(train_X), vec(train_y.Type))
#1737.969209 seconds (218.93 M allocations: 78.189 GiB, 24.12% gc time, 0.05% compilation time)
#RandomForestClassifier
#n_trees:             200
#n_subfeatures:       -1
#partial_sampling:    0.7
#max_depth:           30
#min_samples_leaf:    1
#min_samples_split:   5
#min_purity_increase: 0.0
#classes:             ["FP", "TP"]
#ensemble:            Ensemble of Decision Trees
#Trees:      200
#Avg Leaves: 136248.45
#Avg Depth:  30.0




JLD.save("RandomForest.BBMap.n_trees100.max_depth30.min_samples_split2.rng123.jld", "model", model) # save randomforest to disk
# JLD.save("RandomForest.BBMap.n_trees100.max_depth8.min_samples_split2.rng123.jld", "model", model) # save randomforest to disk
# JLD.save("RandomForest.BBMap.n_trees100.max_depth4.min_samples_split2.rng123.jld", "model", model) # save randomforest to disk
# JLD.save("RandomForest.BBMap.n_trees100.max_depth2.min_samples_split2.rng123.jld", "model", model) # save randomforest to disk
# model = JLD.load("RandomForest.BBMap.jld", "model") # Load back the pipeline


# Make predictions on test set
@time predictions = predict_proba(model, Matrix(test_X))[:, 2]
# 139.969395 seconds (4.06 M allocations: 1.366 GiB, 3.32% gc time, 0.03% compilation time)


# Get the predicted class labels
predicted_labels = [if prob[1] >= 0.5 "TP" else "FP" end for prob in predictions]


# Calculate accuracy
accuracy = mean(predicted_labels .== test_y.Type)
println("Accuracy: $accuracy")






# for RandomForest.BBMap.n_trees100.max_depth60min_samples_split5.rng=123.jld
# Accuracy: 0.7989414319797719
```
