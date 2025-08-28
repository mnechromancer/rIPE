"""
Tests for DATA-002: RNA-seq Integration

Tests the RNA-seq import functionality including DESeq2/EdgeR parsing,
gene ID mapping, expression matrix handling, and pathway enrichment integration.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ipe.lab_integration.molecular.rnaseq_import import (
    RNASeqImporter,
    RNASeqExperiment,
    ExpressionData,
    Gene,
    GeneIDMapper,
)
from ipe.lab_integration.molecular.deseq_parser import (
    DESeqParser,
    PathwayAnalysis,
    EnrichmentResult,
    PathwayTerm,
)


class TestGene:
    """Test suite for Gene class"""

    def test_gene_creation(self):
        """Test basic gene creation"""
        gene = Gene(
            gene_id="ENSMUSG00000000001",
            gene_symbol="Gnai3",
            gene_name="G protein subunit alpha i3",
            gene_type="protein_coding",
            chromosome="3",
            start_pos=108107280,
            end_pos=108146146,
        )

        assert gene.gene_id == "ENSMUSG00000000001"
        assert gene.gene_symbol == "Gnai3"
        assert gene.gene_name == "G protein subunit alpha i3"
        assert gene.gene_type == "protein_coding"
        assert gene.chromosome == "3"

    def test_gene_validation(self):
        """Test gene validation"""
        # Test missing gene ID
        with pytest.raises(ValueError, match="Gene ID is required"):
            Gene(gene_id="")


class TestExpressionData:
    """Test suite for ExpressionData class"""

    def create_test_gene(self):
        """Helper to create test gene"""
        return Gene(
            gene_id="ENSMUSG00000000001", gene_symbol="Gnai3", gene_name="test gene"
        )

    def test_expression_data_creation(self):
        """Test expression data creation"""
        gene = self.create_test_gene()
        expr_data = ExpressionData(
            gene=gene, base_mean=100.5, log2_fold_change=1.5, p_value=0.01, p_adj=0.05
        )

        assert expr_data.gene == gene
        assert expr_data.base_mean == 100.5
        assert expr_data.log2_fold_change == 1.5
        assert expr_data.fold_change == pytest.approx(2.83, abs=0.01)  # 2^1.5

    def test_significance_detection(self):
        """Test significance detection"""
        gene = self.create_test_gene()

        # Significant gene
        sig_expr = ExpressionData(
            gene=gene,
            base_mean=100.0,
            log2_fold_change=2.0,  # > 1.0 threshold
            p_adj=0.01,  # < 0.05 threshold
        )
        assert sig_expr.is_significant()

        # Non-significant (high p-value)
        nonsig_expr1 = ExpressionData(
            gene=gene,
            base_mean=100.0,
            log2_fold_change=2.0,
            p_adj=0.1,  # > 0.05 threshold
        )
        assert not nonsig_expr1.is_significant()

        # Non-significant (low fold change)
        nonsig_expr2 = ExpressionData(
            gene=gene,
            base_mean=100.0,
            log2_fold_change=0.5,  # < 1.0 threshold
            p_adj=0.01,
        )
        assert not nonsig_expr2.is_significant()


class TestGeneIDMapper:
    """Test suite for GeneIDMapper"""

    def test_basic_mapping(self):
        """Test basic gene ID mapping"""
        mapper = GeneIDMapper()

        # Test default mappings
        symbol = mapper.get_symbol("ENSMUSG00000000001")
        assert symbol == "Gnai3"

        name = mapper.get_name("ENSMUSG00000000001")
        assert name == "G protein subunit alpha i3"

    def test_add_mapping(self):
        """Test adding custom mappings"""
        mapper = GeneIDMapper()

        # Add new mapping
        mapper.add_mapping("TEST001", "TestGene", "Test gene name")

        assert mapper.get_symbol("TEST001") == "TestGene"
        assert mapper.get_name("TEST001") == "Test gene name"

    def test_load_mapping_file(self):
        """Test loading mappings from file"""
        # Create test mapping file
        temp_dir = tempfile.mkdtemp()
        mapping_file = Path(temp_dir) / "gene_mapping.csv"

        mapping_data = pd.DataFrame(
            {
                "gene_id": ["GENE001", "GENE002", "GENE003"],
                "gene_symbol": ["Symbol1", "Symbol2", "Symbol3"],
                "gene_name": ["Name 1", "Name 2", "Name 3"],
            }
        )
        mapping_data.to_csv(mapping_file, index=False)

        # Load mappings
        mapper = GeneIDMapper()
        mapper.load_mapping_file(mapping_file)

        assert mapper.get_symbol("GENE001") == "Symbol1"
        assert mapper.get_name("GENE002") == "Name 2"


class TestRNASeqExperiment:
    """Test suite for RNASeqExperiment"""

    def create_test_experiment(self, n_genes=10):
        """Helper to create test experiment"""
        expression_data = []

        for i in range(n_genes):
            gene = Gene(
                gene_id=f"GENE{i:03d}",
                gene_symbol=f"Symbol{i}",
                gene_name=f"Gene {i} name",
            )

            # Make some genes significant
            is_sig = i < n_genes // 2
            p_adj = 0.01 if is_sig else 0.1
            lfc = 2.0 if is_sig else 0.5

            expr_data = ExpressionData(
                gene=gene,
                base_mean=100.0 + i * 10,
                log2_fold_change=lfc * (1 if i % 2 == 0 else -1),  # Mix up/down
                p_adj=p_adj,
            )
            expression_data.append(expr_data)

        return RNASeqExperiment(
            experiment_id="test_experiment",
            sample_groups={
                "control": ["ctrl1", "ctrl2", "ctrl3"],
                "treatment": ["trt1", "trt2", "trt3"],
            },
            expression_data=expression_data,
            metadata={"test": "data"},
        )

    def test_experiment_creation(self):
        """Test experiment creation and indexing"""
        experiment = self.create_test_experiment(6)

        assert experiment.experiment_id == "test_experiment"
        assert len(experiment.expression_data) == 6
        assert "control" in experiment.sample_groups
        assert "treatment" in experiment.sample_groups

    def test_gene_lookup(self):
        """Test gene lookup by ID and symbol"""
        experiment = self.create_test_experiment(3)

        # Lookup by ID
        expr_data = experiment.get_expression_by_gene_id("GENE001")
        assert expr_data is not None
        assert expr_data.gene.gene_id == "GENE001"

        # Lookup by symbol
        expr_data = experiment.get_expression_by_symbol("Symbol2")
        assert expr_data is not None
        assert expr_data.gene.gene_symbol == "Symbol2"

        # Non-existent lookup
        assert experiment.get_expression_by_gene_id("NONEXISTENT") is None

    def test_significance_filtering(self):
        """Test significance filtering methods"""
        experiment = self.create_test_experiment(10)

        # Get significant genes (first 5 should be significant)
        sig_genes = experiment.get_significant_genes()
        assert len(sig_genes) == 5

        # Get upregulated (even indices of significant genes)
        up_genes = experiment.get_upregulated_genes()
        assert len(up_genes) == 3  # 0, 2, 4
        assert all(expr.log2_fold_change > 0 for expr in up_genes)

        # Get downregulated (odd indices of significant genes)
        down_genes = experiment.get_downregulated_genes()
        assert len(down_genes) == 2  # 1, 3
        assert all(expr.log2_fold_change < 0 for expr in down_genes)


class TestRNASeqImporter:
    """Test suite for RNASeqImporter"""

    def setup_method(self):
        """Setup test fixtures"""
        self.importer = RNASeqImporter()
        self.temp_dir = tempfile.mkdtemp()

    def create_deseq2_file(self, filename="deseq2_results.csv", n_genes=10):
        """Create test DESeq2 results file"""
        filepath = Path(self.temp_dir) / filename

        data = {
            "gene_id": [f"ENSMUSG{i:08d}" for i in range(n_genes)],
            "baseMean": np.random.uniform(10, 1000, n_genes),
            "log2FoldChange": np.random.normal(0, 1.5, n_genes),
            "lfcSE": np.random.uniform(0.1, 0.5, n_genes),
            "stat": np.random.normal(0, 2, n_genes),
            "pvalue": np.random.uniform(0, 1, n_genes),
            "padj": np.random.uniform(0, 1, n_genes),
        }

        df = pd.DataFrame(data)
        df.set_index("gene_id", inplace=True)
        df.to_csv(filepath)

        return filepath

    def create_edger_file(self, filename="edger_results.tsv", n_genes=10):
        """Create test EdgeR results file"""
        filepath = Path(self.temp_dir) / filename

        data = {
            "gene_id": [f"ENSMUSG{i:08d}" for i in range(n_genes)],
            "logFC": np.random.normal(0, 1.5, n_genes),
            "logCPM": np.random.uniform(1, 10, n_genes),
            "LR": np.random.uniform(0, 20, n_genes),
            "PValue": np.random.uniform(0, 1, n_genes),
            "FDR": np.random.uniform(0, 1, n_genes),
        }

        df = pd.DataFrame(data)
        df.set_index("gene_id", inplace=True)
        df.to_csv(filepath, sep="\t")

        return filepath

    def create_expression_matrix(
        self, filename="expression_matrix.csv", n_genes=10, n_samples=6
    ):
        """Create test expression matrix"""
        filepath = Path(self.temp_dir) / filename

        sample_names = [f"sample_{i}" for i in range(n_samples)]
        gene_names = [f"GENE{i:03d}" for i in range(n_genes)]

        # Generate count data (Poisson-like)
        data = np.random.poisson(100, (n_genes, n_samples))

        df = pd.DataFrame(data, index=gene_names, columns=sample_names)
        df.to_csv(filepath)

        return filepath

    def test_deseq2_import(self):
        """Test DESeq2 file import"""
        filepath = self.create_deseq2_file(n_genes=5)
        experiment = self.importer.import_deseq2_results(filepath, "test_deseq2")

        assert experiment.experiment_id == "test_deseq2"
        assert len(experiment.expression_data) == 5
        assert experiment.metadata["analysis_tool"] == "DESeq2"

        # Check data integrity
        for expr in experiment.expression_data:
            assert expr.gene.gene_id.startswith("ENSMUSG")
            assert expr.base_mean >= 0
            assert -10 <= expr.log2_fold_change <= 10  # Reasonable range

    def test_edger_import(self):
        """Test EdgeR file import"""
        filepath = self.create_edger_file(n_genes=5)
        experiment = self.importer.import_edger_results(filepath, "test_edger")

        assert experiment.experiment_id == "test_edger"
        assert len(experiment.expression_data) == 5
        assert experiment.metadata["analysis_tool"] == "EdgeR"

        # Check data integrity
        for expr in experiment.expression_data:
            assert expr.gene.gene_id.startswith("ENSMUSG")
            assert expr.base_mean >= 0

    def test_expression_matrix_import(self):
        """Test expression matrix import"""
        filepath = self.create_expression_matrix(n_genes=5, n_samples=6)
        sample_groups = {
            "control": ["sample_0", "sample_1", "sample_2"],
            "treatment": ["sample_3", "sample_4", "sample_5"],
        }

        experiment = self.importer.import_expression_matrix(
            filepath, "test_matrix", sample_groups
        )

        assert experiment.experiment_id == "test_matrix"
        assert len(experiment.expression_data) == 5
        assert experiment.sample_groups == sample_groups
        assert experiment.metadata["analysis_tool"] == "Expression Matrix"

        # Check that raw counts are present
        for expr in experiment.expression_data:
            assert expr.raw_counts is not None
            assert len(expr.raw_counts) == 6

    def test_export_csv(self):
        """Test CSV export"""
        filepath = self.create_deseq2_file(n_genes=3)
        experiment = self.importer.import_deseq2_results(filepath, "test_export")

        output_path = Path(self.temp_dir) / "exported.csv"
        self.importer.export_experiment(experiment, output_path, format="csv")

        assert output_path.exists()

        # Verify export content
        exported_df = pd.read_csv(output_path)
        assert len(exported_df) == 3
        assert "gene_id" in exported_df.columns
        assert "log2_fold_change" in exported_df.columns

    def test_export_json(self):
        """Test JSON export"""
        filepath = self.create_deseq2_file(n_genes=3)
        experiment = self.importer.import_deseq2_results(filepath, "test_export")

        output_path = Path(self.temp_dir) / "exported.json"
        self.importer.export_experiment(experiment, output_path, format="json")

        assert output_path.exists()

        # Verify export content
        import json

        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["experiment_id"] == "test_export"
        assert len(data["genes"]) == 3
        assert "summary" in data


class TestDESeqParser:
    """Test suite for DESeqParser"""

    def setup_method(self):
        """Setup test fixtures"""
        self.parser = DESeqParser()
        self.temp_dir = tempfile.mkdtemp()

    def create_gsea_results(self, filename="gsea_results.tsv"):
        """Create test GSEA results file"""
        filepath = Path(self.temp_dir) / filename

        data = {
            "NAME": ["PATHWAY_1", "PATHWAY_2", "PATHWAY_3"],
            "GS DETAILS": ["Description 1", "Description 2", "Description 3"],
            "SIZE": [50, 75, 100],
            "ES": [0.5, -0.3, 0.7],
            "NES": [1.8, -1.5, 2.1],
            "NOM p-val": [0.01, 0.05, 0.001],
            "FDR q-val": [0.05, 0.1, 0.01],
            "LEADING EDGE": [
                "Gene1,Gene2,Gene3",
                "Gene4,Gene5",
                "Gene6,Gene7,Gene8,Gene9",
            ],
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, sep="\t", index=False)

        return filepath

    def create_david_results(self, filename="david_results.tsv"):
        """Create test DAVID results file"""
        filepath = Path(self.temp_dir) / filename

        data = {
            "Category": ["GO_BP", "KEGG_PATHWAY", "GO_MF"],
            "Term": [
                "GO:0006412~translation",
                "mmu04110:Cell cycle",
                "GO:0003824~catalytic activity",
            ],
            "Count": [15, 8, 20],
            "PValue": [0.001, 0.01, 0.005],
            "Genes": ["Gene1,Gene2,Gene3", "Gene4,Gene5", "Gene6,Gene7,Gene8"],
            "Fold Enrichment": [3.5, 2.1, 1.8],
            "Benjamini": [0.01, 0.05, 0.02],
        }

        df = pd.DataFrame(data)
        df.to_csv(filepath, sep="\t", index=False)

        return filepath

    def test_gsea_parsing(self):
        """Test GSEA results parsing"""
        filepath = self.create_gsea_results()
        pathway_analysis = self.parser.parse_gsea_results(filepath, "test_gsea")

        assert pathway_analysis.analysis_id == "test_gsea"
        assert len(pathway_analysis.enrichment_results) == 3
        assert pathway_analysis.metadata["analysis_tool"] == "GSEA"

        # Check significant pathways (p_adj < 0.05)
        sig_pathways = pathway_analysis.get_significant_pathways()
        assert len(sig_pathways) == 1  # Third pathway (p_adj=0.01)

    def test_david_parsing(self):
        """Test DAVID results parsing"""
        filepath = self.create_david_results()
        pathway_analysis = self.parser.parse_david_results(filepath, "test_david")

        assert pathway_analysis.analysis_id == "test_david"
        assert len(pathway_analysis.enrichment_results) == 3
        assert pathway_analysis.metadata["analysis_tool"] == "DAVID"

        # Check pathway categories
        bp_pathways = pathway_analysis.get_pathways_by_category("GO_BP")
        assert len(bp_pathways) == 1
        assert bp_pathways[0].term.term_name == "GO:0006412~translation"

    def test_contrast_comparison(self):
        """Test multiple contrast comparison"""
        # Create mock experiments
        from ipe.lab_integration.molecular.rnaseq_import import (
            Gene,
            ExpressionData,
            RNASeqExperiment,
        )

        # Create genes for testing
        genes = [Gene(gene_id=f"GENE{i}", gene_symbol=f"Symbol{i}") for i in range(10)]

        # Experiment 1: genes 0-4 significant
        expr_data_1 = []
        for i, gene in enumerate(genes):
            p_adj = 0.01 if i < 5 else 0.1
            lfc = 2.0 if i < 5 else 0.5
            expr_data_1.append(ExpressionData(gene, 100.0, lfc, p_adj=p_adj))

        exp1 = RNASeqExperiment("exp1", {}, expr_data_1, {})

        # Experiment 2: genes 3-7 significant (overlap with exp1)
        expr_data_2 = []
        for i, gene in enumerate(genes):
            p_adj = 0.01 if 3 <= i < 8 else 0.1
            lfc = 2.0 if 3 <= i < 8 else 0.5
            expr_data_2.append(ExpressionData(gene, 100.0, lfc, p_adj=p_adj))

        exp2 = RNASeqExperiment("exp2", {}, expr_data_2, {})

        # Compare experiments
        comparison = self.parser.compare_contrasts(
            {"contrast1": exp1, "contrast2": exp2}
        )

        assert comparison["summary"]["total_contrasts"] == 2
        assert len(comparison["common_genes"]) == 2  # Genes 3, 4
        assert "contrast1_vs_contrast2" in comparison["pairwise_overlaps"]

        # Check overlap counts
        overlap_data = comparison["pairwise_overlaps"]["contrast1_vs_contrast2"]
        assert overlap_data["overlap_count"] == 2


class TestIntegration:
    """Integration tests for RNA-seq analysis workflow"""

    def test_full_rnaseq_workflow(self):
        """Test complete RNA-seq analysis workflow"""
        temp_dir = tempfile.mkdtemp()

        # Create test DESeq2 file
        deseq_data = {
            "gene_id": ["GENE001", "GENE002", "GENE003", "GENE004"],
            "baseMean": [100.0, 200.0, 50.0, 300.0],
            "log2FoldChange": [2.0, -1.5, 0.5, 2.5],
            "pvalue": [0.001, 0.01, 0.1, 0.005],
            "padj": [0.01, 0.05, 0.2, 0.02],
        }

        df = pd.DataFrame(deseq_data)
        df.set_index("gene_id", inplace=True)
        deseq_file = Path(temp_dir) / "test_deseq.csv"
        df.to_csv(deseq_file)

        # Step 1: Import DESeq2 results
        importer = RNASeqImporter()
        experiment = importer.import_deseq2_results(deseq_file, "integration_test")

        # Step 2: Create pathway analysis
        pathway_terms = [
            PathwayTerm(
                "PATH001",
                "Test Pathway 1",
                "Description 1",
                "TEST",
                10,
                {"GENE001", "GENE002"},
            ),
            PathwayTerm(
                "PATH002",
                "Test Pathway 2",
                "Description 2",
                "TEST",
                15,
                {"GENE003", "GENE004"},
            ),
        ]

        enrichment_results = [
            EnrichmentResult(
                term=pathway_terms[0],
                genes_in_term={"GENE001", "GENE002"},
                p_value=0.01,
                p_adj=0.05,
            ),
            EnrichmentResult(
                term=pathway_terms[1],
                genes_in_term={"GENE003", "GENE004"},
                p_value=0.001,
                p_adj=0.01,
            ),
        ]

        pathway_analysis = PathwayAnalysis(
            analysis_id="integration_pathways",
            input_genes={"GENE001", "GENE002", "GENE003", "GENE004"},
            enrichment_results=enrichment_results,
            metadata={},
        )

        # Step 3: Integrate results
        parser = DESeqParser()
        integration = parser.integrate_expression_pathways(experiment, pathway_analysis)

        # Verify integration
        assert integration["experiment_id"] == "integration_test"
        assert integration["pathway_analysis_id"] == "integration_pathways"
        assert (
            integration["statistics"]["significant_genes"] == 2
        )  # GENE001, GENE004 (padj <= 0.05)
        assert (
            integration["statistics"]["significant_pathways"] == 1
        )  # Only one pathway with p_adj < 0.05
        assert len(integration["pathway_overlaps"]) > 0

        # Step 4: Export results
        output_file = Path(temp_dir) / "integrated_results.json"
        parser.export_integrated_results(integration, output_file)

        assert output_file.exists()

        # Verify export
        import json

        with open(output_file, "r") as f:
            exported_data = json.load(f)

        assert exported_data["experiment_id"] == "integration_test"


@pytest.mark.performance
class TestPerformance:
    """Performance tests for RNA-seq import"""

    def test_large_dataset_import(self):
        """Test import performance with large datasets"""
        import time

        temp_dir = tempfile.mkdtemp()

        # Create large DESeq2 file (10,000 genes)
        n_genes = 10000
        data = {
            "gene_id": [f"ENSMUSG{i:08d}" for i in range(n_genes)],
            "baseMean": np.random.uniform(10, 1000, n_genes),
            "log2FoldChange": np.random.normal(0, 1.5, n_genes),
            "pvalue": np.random.uniform(0, 1, n_genes),
            "padj": np.random.uniform(0, 1, n_genes),
        }

        df = pd.DataFrame(data)
        df.set_index("gene_id", inplace=True)
        large_file = Path(temp_dir) / "large_deseq.csv"
        df.to_csv(large_file)

        # Time the import
        importer = RNASeqImporter()
        start_time = time.time()
        experiment = importer.import_deseq2_results(large_file, "performance_test")
        import_time = time.time() - start_time

        # Verify performance and correctness
        assert len(experiment.expression_data) == n_genes
        assert import_time < 10.0  # Should complete within 10 seconds
        assert experiment.experiment_id == "performance_test"
