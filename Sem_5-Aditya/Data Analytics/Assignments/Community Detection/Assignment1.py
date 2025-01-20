from extract import import_wiki_vote_data, import_lastfm_asia_data
from girvan import girvan_newman_one_level, girvan_newman
from louvain import louvain_one_iter
from dendro import visualise_dendrogram

if __name__ == "__main__":
    nodes_connectivity_list_wiki = import_wiki_vote_data("data/Wiki-Vote.txt")
    graph_partition_wiki, _  = girvan_newman_one_level(edges=nodes_connectivity_list_wiki)
    community_mat_wiki = girvan_newman(nodes_connectivity_list_wiki)

    visualise_dendrogram(community_mat_wiki)

    graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki)


    nodes_connectivity_list_lastfm = import_lastfm_asia_data("data/lastfm_asia_edges.csv")
    graph_partition_lastfm, _ = girvan_newman_one_level(edges=nodes_connectivity_list_lastfm)
    community_mat_lastfm = girvan_newman(nodes_connectivity_list_lastfm)

    visualise_dendrogram(community_mat_lastfm)

    graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm)