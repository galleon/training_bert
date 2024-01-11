import csv
import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import networkx as nx
import json

file_path = "data/public_SimilarityAnalytics_export_2023-09-11_161902.csv"
output_path = "data/similarity_data.csv"

outputf = open(output_path, "w")
writer = csv.writer(outputf)

with open(file_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        sourceid = row[1]
        targetid = row[2]
        notation = row[4]
        nrow = [sourceid, targetid, notation]
        if notation != "-1":
            writer.writerow(nrow)
outputf.close()


df_similarity = pd.read_csv("data/similarity_data.csv")

df = pd.read_csv("data/videos.csv")

# dropna lines
df_clean = df.dropna()

# define the pattern to define tags
# pattern = r"\((\w+(?:-\w+)*)\)|(\w+(?:-\w+)*)"

# Define stop words
stop_words = ["oui"]  # Add your stop words here
# define the minumn number of characrers for a tag
min_char = 3


# Define the extract_tags function
def extract_tags(tags):
    # tags = re.findall(pattern, tags)
    # # Flatten the list and remove empty strings
    # tags = [tag for tag in sum(tags, ()) if tag]

    tags = tags.lower().split()  # Convert to lowercase and split
    return [
        tag.strip("()")
        for tag in tags
        if len(tag) >= min_char and tag not in stop_words
    ]


# Apply the extract_tags function to the 'tags' column
df_clean["extracted_tags"] = df_clean["tags"].apply(extract_tags)

# Flatten the list of extracted tags
all_extracted_tags = [
    tag for tags_list in df_clean["extracted_tags"] for tag in tags_list
]

# Count extracted tag occurrences
tag_counts = Counter(all_extracted_tags)

# Filter extracted tags based on criteria
min_count = 500  # Define the minimum count threshold
filtered_tags = [
    tag
    for tag, count in tag_counts.items()
    if count >= min_count and tag not in stop_words
]

# Update the 'extracted_tags' column to include only filtered tags
df_clean.loc[:, "extracted_tags"] = df_clean["extracted_tags"].apply(
    lambda tags: [tag for tag in tags if tag in filtered_tags]
)


# Sort tags by occurrences in descending order
sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

# Print the top 20 tags
top_50_tags = sorted_tags[:50]
for tag, count in top_50_tags:
    print(f"{tag.ljust(20)}\t{count}")

tag_names, tag_occurrences = zip(*top_50_tags)

plt.figure(figsize=(12, 6))
plt.bar(tag_names, tag_occurrences)
plt.xlabel("Tag")
plt.ylabel("Count")
plt.title(f"Top 50 Tags in {len(df_clean)} Videos (min. count: {min(tag_occurrences)})")
plt.xticks(rotation=90)
plt.savefig("data/tags.png")


G = nx.Graph()

# # Add nodes being the ids of the videos
# G.add_nodes_from(df_clean["id"])

# min_shared_tags = 1_000_000  # Define the minimum number of shared tags
# max_shared_tags = 0  # Define the maximum number of shared tags

# # Add edges between videos that share at least one tag
# for _, row1 in df_clean.iterrows():
#     tags_1 = row1["extracted_tags"]
#     video_1 = row1["id"]

#     for _, row2 in df_clean.iterrows():
#         tags_2 = row2["extracted_tags"]
#         video_2 = row2["id"]

#         # Check if the two videos share at least one tag
#         number_of_shared_tags = len(set(tags_1) & set(tags_2))

#         if number_of_shared_tags < min_shared_tags:
#             min_shared_tags = number_of_shared_tags
#         if number_of_shared_tags > max_shared_tags:
#             max_shared_tags = number_of_shared_tags

#         print(video_1, video_2, number_of_shared_tags)

#         if number_of_shared_tags > 0:
#             G.add_edge(video_1, video_2, weight=number_of_shared_tags)

# print(f"Min. shared tags: {min_shared_tags}")
# print(f"Max. shared tags: {max_shared_tags}")

# pos = nx.spring_layout(G, k=0.1)

# nx.draw(G, pos, node_size=10, node_color="blue", alpha=0.5, with_labels=False)
# plt.savefig("data/graph.png")


G = nx.DiGraph()

# Add nodes (SessionIds) to the graph
G.add_nodes_from(df_similarity["SourceSessionId"])
G.add_nodes_from(df_similarity["SimilaritySessionId"])

# Add edges between nodes (SessionIds) based on similarity
for _, row in df_similarity.iterrows():
    source_session_id = row["SourceSessionId"]
    similarity_session_id = row["SimilaritySessionId"]
    similarity = row["Notation"]

    # Add an edge between nodes if similarity is above a threshold (e.g., 0.8)
    if similarity > 0.8:
        G.add_edge(source_session_id, similarity_session_id, weight=similarity)

# Export the graph to JSON including edge weights
graph_data = {
    "nodes": [{"id": int(node)} for node in G.nodes()],
    "links": [
        {
            "source": int(source),
            "target": int(target),
            "weight": int(G[source][target].get("weight", 1)),
            "provenannce": 1,
        }
        for source, target in G.edges()
    ],
}


# Check the data types in the 'graph_data' dictionary
def check_data_types(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                check_data_types(value)
            elif not isinstance(value, (int, float, str, bool, type(None))):
                print(
                    f"Data type error: '{key}' has an unsupported data type: {type(value)}"
                )
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, dict)):
                check_data_types(item)
            elif not isinstance(item, (int, float, str, bool, type(None))):
                print(
                    f"Data type error: List item has an unsupported data type: {type(item)}"
                )


# Call the function to check data types in 'graph_data'
check_data_types(graph_data)

with open("data/graph_data_1.json", "w") as json_file:
    json.dump(graph_data, json_file)


G_csv = nx.DiGraph()

all_similarity_session_ids = df_similarity["SimilaritySessionId"].to_list()

for _, row in df_similarity.iterrows():
    source_session_id = row["SourceSessionId"]
    similarity_session_id = row["SimilaritySessionId"]
    similarity = row["Notation"]

    # check if source_session_id is in the df_clean
    if source_session_id in df_clean["id"].to_list():
        G_csv.add_node(source_session_id)

        # extract the tags from the source session id
        source_tags = df_clean[df_clean["id"] == source_session_id][
            "extracted_tags"
        ].to_list()[0]

        for _, row2 in df_clean.iterrows():
            target_session_id = row2["id"]

            if (
                target_session_id != source_session_id
                and target_session_id in all_similarity_session_ids
            ):
                target_tags = row2["extracted_tags"]

                # if the target_tags list is non empty
                if target_tags:
                    # compute the similarity between the source and target tags
                    target_similarity = len(set(source_tags) & set(target_tags)) / len(
                        set(source_tags) | set(target_tags)
                    )

                    # Add an edge between nodes if similarity is above a threshold (e.g., 0.8)
                    if target_similarity > 0.5:
                        G_csv.add_node(target_session_id)
                        G_csv.add_edge(
                            source_session_id,
                            target_session_id,
                            weight=1,
                        )
    else:
        print(f"{source_session_id} not in df_clean")

# Export the graph to JSON including edge weights
graph_data = {
    "nodes": [{"id": int(node)} for node in G_csv.nodes()],
    "links": [
        {
            "source": int(source),
            "target": int(target),
            "weight": int(G_csv[source][target].get("weight", 1)),
            "provenannce": 2,
        }
        for source, target in G_csv.edges()
    ],
}

# Export the graph to JSON including edge weights
with open("data/graph_data_2.json", "w") as json_file:
    json.dump(graph_data, json_file)
