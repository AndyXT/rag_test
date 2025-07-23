#!/usr/bin/env python3
"""Test and improve retrieval relevance"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_retrieval(query, vectorstore, k=5):
    """Test retrieval with different methods"""
    console.print(f"\n[bold]Testing query:[/bold] '{query}'")
    console.print(f"[dim]Retrieving top {k} documents...[/dim]\n")
    
    # Method 1: Similarity search with scores
    console.print("[bold cyan]Method 1: Similarity Search with Scores[/bold cyan]")
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    table = Table(title="Retrieved Documents with Relevance Scores")
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="yellow")
    table.add_column("Content Preview", style="white", max_width=60)
    table.add_column("Metadata", style="dim")
    
    for i, (doc, score) in enumerate(docs_with_scores):
        preview = doc.page_content[:150].replace('\n', ' ')
        metadata = str(doc.metadata.get('source', 'Unknown'))[:30]
        table.add_row(
            str(i+1),
            f"{score:.3f}",
            preview + "...",
            metadata
        )
    
    console.print(table)
    
    # Method 2: MMR (Maximum Marginal Relevance) for diversity
    console.print("\n[bold cyan]Method 2: MMR Search (Better Diversity)[/bold cyan]")
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
    
    table2 = Table(title="MMR Retrieved Documents")
    table2.add_column("Rank", style="cyan")
    table2.add_column("Content Preview", style="white", max_width=80)
    
    for i, doc in enumerate(mmr_docs):
        preview = doc.page_content[:150].replace('\n', ' ')
        table2.add_row(str(i+1), preview + "...")
    
    console.print(table2)
    
    return docs_with_scores

def analyze_retrieval_quality(query, vectorstore):
    """Analyze why retrieval might not be working well"""
    console.print(Panel("[bold]Retrieval Quality Analysis[/bold]", expand=False))
    
    # Get some documents
    docs = vectorstore.similarity_search_with_score(query, k=10)
    
    if not docs:
        console.print("[red]❌ No documents retrieved![/red]")
        return
    
    scores = [score for _, score in docs]
    
    # Analyze score distribution
    console.print("\n[bold]Score Statistics:[/bold]")
    console.print(f"  Best score: {min(scores):.3f} (lower is better)")
    console.print(f"  Worst score: {max(scores):.3f}")
    console.print(f"  Average: {sum(scores)/len(scores):.3f}")
    console.print(f"  Spread: {max(scores) - min(scores):.3f}")
    
    # Check if scores are too similar (poor discrimination)
    if max(scores) - min(scores) < 0.1:
        console.print("\n[yellow]⚠️  Scores are very similar - poor discrimination[/yellow]")
        console.print("   This suggests:")
        console.print("   • Chunks may be too large/generic")
        console.print("   • Query might be too vague")
        console.print("   • Consider smaller chunks or better embedding model")
    
    # Check if all scores are high (poor matches)
    if min(scores) > 0.5:
        console.print("\n[yellow]⚠️  All scores are high - poor matches overall[/yellow]")
        console.print("   This suggests:")
        console.print("   • Query doesn't match document content well")
        console.print("   • Consider rephrasing query")
        console.print("   • Check if documents contain expected content")

def suggest_improvements(vectorstore):
    """Suggest improvements based on current setup"""
    console.print(Panel("[bold]Improvement Suggestions[/bold]", expand=False))
    
    # Get collection stats
    collection = vectorstore._collection
    count = collection.count()
    
    console.print(f"\n[bold]Current Setup:[/bold]")
    console.print(f"  Total chunks: {count}")
    
    console.print("\n[bold]Recommendations:[/bold]")
    
    if count > 10000:
        console.print("  • [yellow]Large corpus detected[/yellow]")
        console.print("    - Consider using a better embedding model")
        console.print("    - Try: sentence-transformers/all-mpnet-base-v2")
    
    console.print("\n  • [cyan]For better relevance:[/cyan]")
    console.print("    1. Reduce chunk_size to 300-500 chars")
    console.print("    2. Increase chunk_overlap to 50-100 chars")
    console.print("    3. Use more specific queries")
    console.print("    4. Consider hybrid search (keyword + semantic)")
    
    console.print("\n  • [cyan]Alternative embedding models:[/cyan]")
    console.print("    - all-mpnet-base-v2 (better quality, larger)")
    console.print("    - all-MiniLM-L12-v2 (good balance)")
    console.print("    - instructor-base (follows instructions)")

def main():
    console.print(Panel.fit("🔍 [bold]RAG Retrieval Tester[/bold]", border_style="blue"))
    
    # Initialize
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Test queries
    test_queries = [
        "What is this document about?",
        "machine learning",
        "python programming",
        "How do I",
    ]
    
    # Interactive mode
    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("1. Test a query")
        console.print("2. Analyze retrieval quality")
        console.print("3. Show improvement suggestions")
        console.print("4. Test predefined queries")
        console.print("5. Exit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == "1":
            query = input("Enter query: ")
            test_retrieval(query, vectorstore)
        elif choice == "2":
            query = input("Enter query to analyze: ")
            analyze_retrieval_quality(query, vectorstore)
        elif choice == "3":
            suggest_improvements(vectorstore)
        elif choice == "4":
            for query in test_queries:
                test_retrieval(query, vectorstore, k=3)
                input("\nPress Enter to continue...")
        elif choice == "5":
            break

if __name__ == "__main__":
    main()