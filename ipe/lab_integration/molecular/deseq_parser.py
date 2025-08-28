"""
DATA-002: RNA-seq Integration - DESeq Parser

This module provides specialized parsing functionality for DESeq2 results
and pathway enrichment analysis integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Any
import pandas as pd
from datetime import datetime

from .rnaseq_import import RNASeqExperiment


@dataclass
class PathwayTerm:
    """Represents a pathway or gene set term"""

    term_id: str
    term_name: str
    description: str
    category: str  # GO_BP, GO_MF, GO_CC, KEGG, etc.
    gene_count: int
    genes: Set[str] = field(default_factory=set)


@dataclass
class EnrichmentResult:
    """Represents pathway enrichment analysis result"""

    term: PathwayTerm
    genes_in_term: Set[str]  # Genes from input that are in this term
    p_value: float
    p_adj: float
    enrichment_score: Optional[float] = None
    fold_enrichment: Optional[float] = None

    def is_significant(self, p_threshold: float = 0.05) -> bool:
        """Check if enrichment is significant"""
        return self.p_adj < p_threshold


@dataclass
class PathwayAnalysis:
    """Contains pathway enrichment analysis results"""

    analysis_id: str
    input_genes: Set[str]
    enrichment_results: List[EnrichmentResult]
    metadata: Dict[str, Any]
    created_date: datetime = field(default_factory=datetime.now)

    def get_significant_pathways(
        self, p_threshold: float = 0.05
    ) -> List[EnrichmentResult]:
        """Get significantly enriched pathways"""
        return [
            result
            for result in self.enrichment_results
            if result.is_significant(p_threshold)
        ]

    def get_pathways_by_category(self, category: str) -> List[EnrichmentResult]:
        """Get pathways from specific category (GO_BP, KEGG, etc.)"""
        return [
            result
            for result in self.enrichment_results
            if result.term.category == category
        ]


class DESeqParser:
    """
    Specialized parser for DESeq2 analysis results and related tools

    Handles:
    - DESeq2 differential expression results
    - Pathway enrichment analysis results (GSEA, topGO, etc.)
    - Gene ontology annotations
    - Multi-contrast comparisons
    """

    def __init__(self):
        """Initialize DESeq parser"""
        self.pathway_databases = {}
        self._load_pathway_databases()

    def _load_pathway_databases(self):
        """Load pathway database information"""
        # In a real implementation, this would load from databases
        # For now, include some example pathways
        self.pathway_databases = {
            "GO:0006412": PathwayTerm(
                term_id="GO:0006412",
                term_name="translation",
                description="The cellular metabolic process in which a protein is formed",
                category="GO_BP",
                gene_count=500,
                genes={"Eef1a1", "Rpl1", "Rps1", "Rpl2", "Rps2"},
            ),
            "GO:0005739": PathwayTerm(
                term_id="GO:0005739",
                term_name="mitochondrion",
                description="A semiautonomous, self replicating organelle",
                category="GO_CC",
                gene_count=1200,
                genes={"Cox1", "Cox2", "Atp5a1", "Ndufa1", "Cytb"},
            ),
            "mmu04110": PathwayTerm(
                term_id="mmu04110",
                term_name="Cell cycle",
                description="KEGG pathway for cell cycle regulation",
                category="KEGG",
                gene_count=124,
                genes={"Cdk1", "Cdk2", "Ccna2", "Ccnb1", "Rb1"},
            ),
        }

    def parse_deseq2_contrasts(
        self, results_dir: Union[str, Path], experiment_id: str
    ) -> Dict[str, RNASeqExperiment]:
        """
        Parse multiple DESeq2 contrast results from a directory

        Args:
            results_dir: Directory containing DESeq2 results files
            experiment_id: Base experiment ID

        Returns:
            Dictionary mapping contrast names to RNASeqExperiment objects
        """
        results_dir = Path(results_dir)
        experiments = {}

        # Find all CSV/TSV files in directory
        for filepath in results_dir.glob("*.csv"):
            contrast_name = filepath.stem

            try:
                from .rnaseq_import import RNASeqImporter

                importer = RNASeqImporter()
                experiment = importer.import_deseq2_results(
                    filepath, f"{experiment_id}_{contrast_name}"
                )
                experiments[contrast_name] = experiment
            except Exception as e:
                print(f"Warning: Failed to parse {filepath}: {e}")

        # Also check for TSV files
        for filepath in results_dir.glob("*.tsv"):
            contrast_name = filepath.stem
            if contrast_name not in experiments:  # Don't overwrite CSV
                try:
                    from .rnaseq_import import RNASeqImporter

                    importer = RNASeqImporter()
                    experiment = importer.import_deseq2_results(
                        filepath, f"{experiment_id}_{contrast_name}"
                    )
                    experiments[contrast_name] = experiment
                except Exception as e:
                    print(f"Warning: Failed to parse {filepath}: {e}")

        return experiments

    def parse_gsea_results(
        self, filepath: Union[str, Path], analysis_id: str
    ) -> PathwayAnalysis:
        """
        Parse Gene Set Enrichment Analysis (GSEA) results

        Expected format:
        - NAME: pathway name
        - GS<br> DETAILS: gene set details
        - SIZE: gene set size
        - ES: enrichment score
        - NES: normalized enrichment score
        - NOM p-val: nominal p-value
        - FDR q-val: FDR adjusted p-value
        - FWER p-val: family-wise error rate
        - RANK AT MAX: rank at maximum ES
        - LEADING EDGE: core enrichment genes

        Args:
            filepath: Path to GSEA results file
            analysis_id: Analysis identifier

        Returns:
            PathwayAnalysis object
        """
        df = pd.read_csv(filepath, sep="	")

        enrichment_results = []
        input_genes = set()

        for _, row in df.iterrows():
            # Parse pathway term
            term_name = row.get("NAME", "")
            term_id = term_name  # GSEA often uses name as ID

            term = PathwayTerm(
                term_id=term_id,
                term_name=term_name,
                description=row.get("GS DETAILS", ""),
                category="GSEA",
                gene_count=int(row.get("SIZE", 0)),
            )

            # Parse enrichment results
            leading_edge = row.get("LEADING EDGE", "")
            genes_in_term = set(leading_edge.split(",")) if leading_edge else set()
            input_genes.update(genes_in_term)

            enrichment_result = EnrichmentResult(
                term=term,
                genes_in_term=genes_in_term,
                p_value=float(row.get("NOM p-val", 1.0)),
                p_adj=float(row.get("FDR q-val", 1.0)),
                enrichment_score=float(row.get("ES", 0.0)),
                fold_enrichment=float(
                    row.get("NES", 0.0)
                ),  # Use NES as fold enrichment
            )

            enrichment_results.append(enrichment_result)

        return PathwayAnalysis(
            analysis_id=analysis_id,
            input_genes=input_genes,
            enrichment_results=enrichment_results,
            metadata={
                "source_file": str(filepath),
                "analysis_tool": "GSEA",
                "import_date": datetime.now().isoformat(),
            },
        )

    def parse_david_results(
        self, filepath: Union[str, Path], analysis_id: str
    ) -> PathwayAnalysis:
        """
        Parse DAVID functional annotation results

        Expected format:
        - Category: annotation category
        - Term: pathway term
        - Count: number of genes
        - %: percentage
        - PValue: p-value
        - Genes: gene list
        - List Total: total genes in list
        - Pop Hits: population hits
        - Pop Total: population total
        - Fold Enrichment: fold enrichment
        - Bonferroni: Bonferroni correction
        - Benjamini: Benjamini-Hochberg FDR
        - FDR: false discovery rate

        Args:
            filepath: Path to DAVID results file
            analysis_id: Analysis identifier

        Returns:
            PathwayAnalysis object
        """
        df = pd.read_csv(filepath, sep="	")

        enrichment_results = []
        input_genes = set()

        for _, row in df.iterrows():
            # Parse pathway term
            category = row.get("Category", "Unknown")
            term_name = row.get("Term", "")

            # Extract term ID from term name (often in format "ID:Name")
            if ":" in term_name:
                term_id = term_name.split(":")[0]
            else:
                term_id = term_name

            term = PathwayTerm(
                term_id=term_id,
                term_name=term_name,
                description=term_name,
                category=category,
                gene_count=int(row.get("Pop Hits", 0)),
            )

            # Parse gene list
            genes_str = row.get("Genes", "")
            genes_in_term = set()
            if genes_str:
                # DAVID often uses comma-separated gene symbols
                genes_in_term = set(gene.strip() for gene in genes_str.split(","))
                input_genes.update(genes_in_term)

            enrichment_result = EnrichmentResult(
                term=term,
                genes_in_term=genes_in_term,
                p_value=float(row.get("PValue", 1.0)),
                p_adj=float(row.get("Benjamini", 1.0)),
                fold_enrichment=float(row.get("Fold Enrichment", 1.0)),
            )

            enrichment_results.append(enrichment_result)

        return PathwayAnalysis(
            analysis_id=analysis_id,
            input_genes=input_genes,
            enrichment_results=enrichment_results,
            metadata={
                "source_file": str(filepath),
                "analysis_tool": "DAVID",
                "import_date": datetime.now().isoformat(),
            },
        )

    def parse_topgo_results(
        self, filepath: Union[str, Path], analysis_id: str, category: str = "GO_BP"
    ) -> PathwayAnalysis:
        """
        Parse topGO enrichment analysis results

        Expected format:
        - GO.ID: Gene Ontology term ID
        - Term: GO term name
        - Annotated: genes annotated to this term
        - Significant: significant genes in this term
        - Expected: expected number of significant genes
        - classicFisher: classic Fisher exact test p-value
        - (additional columns for other algorithms)

        Args:
            filepath: Path to topGO results file
            analysis_id: Analysis identifier
            category: GO category (GO_BP, GO_MF, GO_CC)

        Returns:
            PathwayAnalysis object
        """
        df = pd.read_csv(filepath, sep="	")

        enrichment_results = []
        input_genes = set()

        for _, row in df.iterrows():
            # Parse GO term
            term_id = row.get("GO.ID", "")
            term_name = row.get("Term", "")

            term = PathwayTerm(
                term_id=term_id,
                term_name=term_name,
                description=term_name,
                category=category,
                gene_count=int(row.get("Annotated", 0)),
            )

            # For topGO, we don't always have the gene list in results
            # In a real implementation, this would be retrieved from GO annotations
            genes_in_term = set()

            # Use classicFisher p-value as primary, fall back to other methods
            p_value = row.get("classicFisher", row.get("weight01", 1.0))

            enrichment_result = EnrichmentResult(
                term=term,
                genes_in_term=genes_in_term,
                p_value=float(p_value),
                p_adj=float(p_value),  # topGO doesn't always provide adjusted p-values
                fold_enrichment=float(row.get("Significant", 0))
                / float(row.get("Expected", 1)),
            )

            enrichment_results.append(enrichment_result)

        return PathwayAnalysis(
            analysis_id=analysis_id,
            input_genes=input_genes,
            enrichment_results=enrichment_results,
            metadata={
                "source_file": str(filepath),
                "analysis_tool": "topGO",
                "go_category": category,
                "import_date": datetime.now().isoformat(),
            },
        )

    def integrate_expression_pathways(
        self,
        experiment: RNASeqExperiment,
        pathway_analysis: PathwayAnalysis,
        p_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Integrate differential expression and pathway enrichment results

        Args:
            experiment: RNA-seq experiment results
            pathway_analysis: Pathway enrichment analysis
            p_threshold: Significance threshold

        Returns:
            Integrated analysis summary
        """
        # Get significant genes from expression analysis
        sig_genes = experiment.get_significant_genes(p_threshold)
        sig_gene_ids = {
            expr.gene.gene_symbol or expr.gene.gene_id for expr in sig_genes
        }

        # Get significant pathways
        sig_pathways = pathway_analysis.get_significant_pathways(p_threshold)

        # Find overlaps between expression and pathway genes
        pathway_gene_overlaps = {}
        for pathway in sig_pathways:
            overlap = sig_gene_ids.intersection(pathway.genes_in_term)
            if overlap:
                pathway_gene_overlaps[pathway.term.term_name] = {
                    "pathway_genes": pathway.genes_in_term,
                    "significant_genes": overlap,
                    "overlap_count": len(overlap),
                    "p_value": pathway.p_value,
                    "p_adj": pathway.p_adj,
                }

        # Create summary
        summary = {
            "experiment_id": experiment.experiment_id,
            "pathway_analysis_id": pathway_analysis.analysis_id,
            "integration_date": datetime.now().isoformat(),
            "statistics": {
                "total_genes_tested": len(experiment.expression_data),
                "significant_genes": len(sig_genes),
                "upregulated_genes": len(experiment.get_upregulated_genes(p_threshold)),
                "downregulated_genes": len(
                    experiment.get_downregulated_genes(p_threshold)
                ),
                "total_pathways_tested": len(pathway_analysis.enrichment_results),
                "significant_pathways": len(sig_pathways),
                "pathways_with_de_genes": len(pathway_gene_overlaps),
            },
            "pathway_overlaps": pathway_gene_overlaps,
        }

        return summary

    def export_integrated_results(
        self,
        integration_summary: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json",
    ):
        """
        Export integrated expression and pathway results

        Args:
            integration_summary: Summary from integrate_expression_pathways
            output_path: Output file path
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_path)

        if format.lower() == "json":
            import json

            with open(output_path, "w") as f:
                json.dump(integration_summary, f, indent=2, default=str)

        elif format.lower() == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(
                    [
                        "pathway_name",
                        "pathway_p_value",
                        "pathway_p_adj",
                        "total_pathway_genes",
                        "significant_de_genes",
                        "overlap_genes",
                    ]
                )

                # Write pathway data
                for pathway_name, data in integration_summary.get(
                    "pathway_overlaps", {}
                ).items():
                    writer.writerow(
                        [
                            pathway_name,
                            f"{data['p_value']:.2e}",
                            f"{data['p_adj']:.2e}",
                            len(data["pathway_genes"]),
                            data["overlap_count"],
                            ";".join(sorted(data["significant_genes"])),
                        ]
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def compare_contrasts(
        self,
        experiments: Dict[str, RNASeqExperiment],
        p_threshold: float = 0.05,
        lfc_threshold: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compare multiple contrasts to find common and unique DE genes

        Args:
            experiments: Dictionary of contrast_name -> RNASeqExperiment
            p_threshold: Significance threshold for p-value
            lfc_threshold: Significance threshold for log fold change

        Returns:
            Comparison summary with overlaps and unique genes
        """
        if not experiments:
            return {}

        # Get significant genes for each contrast
        contrast_genes = {}
        all_genes = set()

        for contrast_name, experiment in experiments.items():
            sig_genes = experiment.get_significant_genes(p_threshold, lfc_threshold)
            gene_ids = {
                expr.gene.gene_symbol or expr.gene.gene_id for expr in sig_genes
            }
            contrast_genes[contrast_name] = gene_ids
            all_genes.update(gene_ids)

        # Find overlaps
        contrast_names = list(contrast_genes.keys())

        # Pairwise overlaps
        pairwise_overlaps = {}
        for i, contrast1 in enumerate(contrast_names):
            for j in range(i + 1, len(contrast_names)):
                contrast2 = contrast_names[j]
                overlap = contrast_genes[contrast1].intersection(
                    contrast_genes[contrast2]
                )
                pair_name = f"{contrast1}_vs_{contrast2}"
                pairwise_overlaps[pair_name] = {
                    "overlap_genes": overlap,
                    "overlap_count": len(overlap),
                    "contrast1_unique": contrast_genes[contrast1]
                    - contrast_genes[contrast2],
                    "contrast2_unique": contrast_genes[contrast2]
                    - contrast_genes[contrast1],
                }

        # Find genes common to all contrasts
        common_genes = (
            set.intersection(*contrast_genes.values())
            if contrast_genes.values()
            else set()
        )

        # Find genes unique to each contrast
        unique_genes = {}
        for contrast_name, genes in contrast_genes.items():
            other_genes = set()
            for other_contrast, other_genes_set in contrast_genes.items():
                if other_contrast != contrast_name:
                    other_genes.update(other_genes_set)
            unique_genes[contrast_name] = genes - other_genes

        return {
            "comparison_date": datetime.now().isoformat(),
            "parameters": {"p_threshold": p_threshold, "lfc_threshold": lfc_threshold},
            "summary": {
                "total_contrasts": len(experiments),
                "total_unique_genes": len(all_genes),
                "common_to_all_contrasts": len(common_genes),
            },
            "contrast_statistics": {
                contrast: {
                    "total_significant": len(genes),
                    "unique_to_contrast": len(unique_genes[contrast]),
                }
                for contrast, genes in contrast_genes.items()
            },
            "pairwise_overlaps": pairwise_overlaps,
            "common_genes": common_genes,
            "unique_genes": unique_genes,
        }
