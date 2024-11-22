from pyspark.sql import SparkSession
from py2neo import Graph, Node, Relationship
import json
from typing import Dict, List, Optional, Set, Tuple
import logging
import builtins
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np

class CitationAnalyzer:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize Citation Analyzer with Neo4j connection"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graph_db = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_citation_graph(self, file_path: str, batch_size: int = 1000):
        """Create citation graph in Neo4j from JSON data"""
        try:
            # Clear existing graph
            self.graph_db.run("MATCH (n) DETACH DELETE n")
            
            # Create constraint for uniqueness
            self.graph_db.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            
            papers_processed = 0
            with open(file_path, 'r') as file:
                tx = self.graph_db.begin()
                current_batch = 0
                
                for line in file:
                    data = json.loads(line)
                    paper_id = data['paper']
                    references = data.get('reference', [])
                    
                    # Create citing paper node
                    citing_paper = Node("Paper", id=paper_id)
                    tx.merge(citing_paper, "Paper", "id")
                    
                    # Create reference nodes and relationships
                    if references:
                        for ref in references:
                            cited_paper = Node("Paper", id=ref)
                            tx.merge(cited_paper, "Paper", "id")
                            cites = Relationship(citing_paper, "CITES", cited_paper)
                            tx.create(cites)
                    
                    current_batch += 1
                    papers_processed += 1
                    
                    if current_batch >= batch_size:
                        tx.commit()
                        tx = self.graph_db.begin()
                        current_batch = 0
                        self.logger.info(f"Processed {papers_processed} papers...")
                
                if current_batch > 0:
                    tx.commit()
                    
            self._log_graph_stats()
            
        except Exception as e:
            self.logger.error(f"Error creating citation graph: {str(e)}")
            raise

    def _log_graph_stats(self):
        """Log statistics about the graph"""
        paper_count = self.graph_db.run("MATCH (p:Paper) RETURN count(p) as count").data()[0]['count']
        cite_count = self.graph_db.run("MATCH ()-[r:CITES]->() RETURN count(r) as count").data()[0]['count']
        self.logger.info(f"Graph statistics - Nodes: {paper_count}, Citations: {cite_count}")

    def get_graph_structure(self) -> Dict[str, List[str]]:
        """Retrieve graph structure from Neo4j"""
        try:
            # Get all nodes first
            all_nodes_query = "MATCH (p:Paper) RETURN p.id as id"
            pred_dict = {record['id']: [] for record in self.graph_db.run(all_nodes_query)}
            
            # Get citation relationships
            citation_query = """
            MATCH (cited:Paper)<-[r:CITES]-(citing:Paper)
            RETURN cited.id as cited, collect(citing.id) as citing_papers
            """
            
            for record in self.graph_db.run(citation_query):
                cited_id = record['cited']
                citing_papers = record['citing_papers']
                pred_dict[cited_id] = citing_papers
                
            return pred_dict
            
        except Exception as e:
            self.logger.error(f"Error retrieving graph structure: {str(e)}")
            raise

    def calculate_simrank(
        self, 
        query_nodes: List[str], 
        C: float = 0.8, 
        max_iterations: int = 100, 
        tolerance: float = 1e-4
    ) -> Dict[str, Dict[str, float]]:
        """Calculate SimRank similarities for query nodes"""
        def calculate_similarity(node1: str, node2: str, prev_sims: Dict[Tuple[str, str], float]) -> float:
            """Calculate similarity between two nodes"""
            if node1 == node2:
                return 1.0
                
            pred1 = graph.get(node1, [])
            pred2 = graph.get(node2, [])
            
            if not pred1 or not pred2:
                return 0.0
                
            sum_sim = sum(
                prev_sims.get((p1, p2), 0.0)
                for p1 in pred1
                for p2 in pred2
            )
            
            return (C / (len(pred1) * len(pred2))) * sum_sim

        try:
            graph = self.get_graph_structure()
            results = {}
            
            for query_node in query_nodes:
                if query_node not in graph:
                    self.logger.warning(f"Query node {query_node} not found in graph")
                    continue
                    
                self.logger.info(f"Processing SimRank for query node: {query_node}")
                
                # Get relevant nodes
                relevant_nodes = {query_node}
                for node, predecessors in graph.items():
                    if predecessors:
                        relevant_nodes.add(node)
                        relevant_nodes.update(predecessors)
                
                # SimRank iterations
                similarities = {}
                prev_similarities = {}
                
                for iteration in range(max_iterations):
                    similarities = {}
                    max_diff = 0.0
                    
                    for node in relevant_nodes:
                        sim = calculate_similarity(query_node, node, prev_similarities)
                        if sim > tolerance:
                            similarities[(query_node, node)] = sim
                            prev_sim = prev_similarities.get((query_node, node), 0.0)
                            max_diff = builtins.max(max_diff, builtins.abs(sim - prev_sim))
                    
                    if max_diff < tolerance:
                        self.logger.info(f"SimRank converged after {iteration + 1} iterations")
                        break
                        
                    prev_similarities = similarities.copy()
                
                # Store results
                node_results = {
                    n2: sim 
                    for (n1, n2), sim in similarities.items() 
                    if n1 == query_node and n2 != query_node and sim > tolerance
                }
                
                results[query_node] = dict(sorted(
                    node_results.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating SimRank: {str(e)}")
            raise

    def print_results(self, results: Dict[str, Dict[str, float]], top_k: int = 10):
        """Print SimRank results"""
        for query_node, similarities in results.items():
            print(f"\nResults for query node: {query_node}")
            print(f"Top {top_k} most similar nodes:")
            
            for node, score in list(similarities.items())[:top_k]:
                print(f"  Node: {node:<15} Similarity: {score:.4f}")

def main():
    # Neo4j configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password123"
    
    # Initialize analyzer
    analyzer = CitationAnalyzer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Create citation graph (if needed)
    file_path = "train.json"
    # analyzer.create_citation_graph(file_path)
    
    # Define query nodes and calculate similarities for different C values
    query_nodes = ["2982615777", "1556418098", "2963981420"]
    
    for c in [0.7, 0.8, 0.9]:
        print(f"\nCalculating SimRank similarities with C = {c}")
        similarities = analyzer.calculate_simrank(query_nodes, C=c)
        analyzer.print_results(similarities)

if __name__ == "__main__":
    main()