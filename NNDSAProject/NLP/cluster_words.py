def cluster_words(graph, num_clusters):
    """
    Improved clustering: Group words into clusters based on connected components using BFS.
    If the number of clusters is greater than the actual number of components,
    clusters are merged. If it's smaller, we split the larger clusters into smaller ones.
    """
    visited = set()  
    clusters = []  

    # Step 1: Perform BFS to find initial connected components (clusters)
    for node in graph.nodes():
        if node not in visited:
            cluster = graph.bfs(node)  # Assuming graph.bfs() returns a cluster of connected words
            clusters.append(cluster)
            visited.update(cluster)

    print("Initial clusters:", clusters)

    # Step 2: Adjust the number of clusters based on num_clusters
    while len(clusters) > num_clusters:
        print(f"Current clusters: {clusters} (Total: {len(clusters)})")

        # If we have more clusters than required, merge the two smallest clusters
        smallest_cluster = min(clusters, key=len)
        clusters.remove(smallest_cluster)

        # Merge the smallest cluster into the first cluster
        clusters[0].extend(smallest_cluster)

    # If we have fewer clusters than required, split larger clusters
    while len(clusters) < num_clusters:
        print(f"Current clusters: {clusters} (Total: {len(clusters)})")

        # Find the largest cluster to split
        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)

        # Split the largest cluster into two smaller clusters
        mid = len(largest_cluster) // 2
        clusters.append(largest_cluster[:mid])
        clusters.append(largest_cluster[mid:])

    print("Final clusters:", clusters)  
    return clusters
