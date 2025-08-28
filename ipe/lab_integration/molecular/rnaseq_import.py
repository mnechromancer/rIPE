"""
DATA-002: RNA-seq Integration

This module implements functionality for importing RNA-seq analysis results
from DESeq2, EdgeR, and other differential expression analysis tools.
Includes gene ID mapping, expression matrix handling, and pathway enrichment integration.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Any
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class Gene:
    """Represents a gene with various identifiers and annotations"""
    gene_id: str
    gene_symbol: Optional[str] = None
    gene_name: Optional[str] = None
    gene_type: Optional[str] = None  # protein_coding, lncRNA, etc.
    chromosome: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    strand: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate gene data"""
        if not self.gene_id:
            raise ValueError("Gene ID is required")


@dataclass 
class ExpressionData:
    """Represents expression data for a single gene"""
    gene: Gene
    base_mean: float
    log2_fold_change: float
    log2_fold_change_se: Optional[float] = None
    stat: Optional[float] = None
    p_value: Optional[float] = None
    p_adj: Optional[float] = None
    raw_counts: Optional[Dict[str, int]] = None  # sample_id -> count
    normalized_counts: Optional[Dict[str, float]] = None  # sample_id -> normalized count
    
    def is_significant(self, p_threshold: float = 0.05, lfc_threshold: float = 1.0) -> bool:
        """Check if gene is significantly differentially expressed"""
        if self.p_adj is None:
            return False
        return (self.p_adj < p_threshold and 
                abs(self.log2_fold_change) > lfc_threshold)
    
    @property
    def fold_change(self) -> float:
        """Calculate linear fold change from log2 fold change"""
        return 2 ** self.log2_fold_change


@dataclass
class RNASeqExperiment:
    """Contains complete RNA-seq experiment data"""
    experiment_id: str
    sample_groups: Dict[str, List[str]]  # group_name -> [sample_ids]
    expression_data: List[ExpressionData]
    metadata: Dict[str, Any]
    created_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Create indices for fast lookup"""
        self._gene_index = {expr.gene.gene_id: expr for expr in self.expression_data}
        self._symbol_index = {expr.gene.gene_symbol: expr for expr in self.expression_data 
                             if expr.gene.gene_symbol}
    
    def get_expression_by_gene_id(self, gene_id: str) -> Optional[ExpressionData]:
        """Get expression data by gene ID"""
        return self._gene_index.get(gene_id)
    
    def get_expression_by_symbol(self, symbol: str) -> Optional[ExpressionData]:
        """Get expression data by gene symbol"""
        return self._symbol_index.get(symbol)
    
    def get_significant_genes(self, p_threshold: float = 0.05, 
                             lfc_threshold: float = 1.0) -> List[ExpressionData]:
        """Get significantly differentially expressed genes"""
        return [expr for expr in self.expression_data 
                if expr.is_significant(p_threshold, lfc_threshold)]
    
    def get_upregulated_genes(self, p_threshold: float = 0.05, 
                             lfc_threshold: float = 1.0) -> List[ExpressionData]:
        """Get significantly upregulated genes"""
        return [expr for expr in self.expression_data 
                if (expr.is_significant(p_threshold, lfc_threshold) and 
                    expr.log2_fold_change > 0)]
    
    def get_downregulated_genes(self, p_threshold: float = 0.05, 
                               lfc_threshold: float = 1.0) -> List[ExpressionData]:
        """Get significantly downregulated genes"""
        return [expr for expr in self.expression_data 
                if (expr.is_significant(p_threshold, lfc_threshold) and 
                    expr.log2_fold_change < 0)]


class GeneIDMapper:
    """
    Maps between different gene identifier systems
    (Ensembl, NCBI, HGNC, etc.)
    """
    
    def __init__(self):
        self.mappings = {}
        self._load_default_mappings()
    
    def _load_default_mappings(self):
        """Load default ID mappings for common organisms"""
        # In a real implementation, this would load from databases or files
        # For now, include some common mouse mappings as examples
        self.mappings = {
            'ENSMUSG00000000001': {'symbol': 'Gnai3', 'name': 'G protein subunit alpha i3'},
            'ENSMUSG00000000028': {'symbol': 'Cdc45', 'name': 'cell division cycle 45'},
            'ENSMUSG00000000037': {'symbol': 'Scml2', 'name': 'sex comb on midleg-like 2'},
            'ENSMUSG00000000056': {'symbol': 'Narf', 'name': 'nuclear prelamin A recognition factor'},
            'ENSMUSG00000000078': {'symbol': 'Klf6', 'name': 'Kruppel like factor 6'},
        }
    
    def add_mapping(self, gene_id: str, symbol: str, name: str = None):
        """Add a gene ID mapping"""
        self.mappings[gene_id] = {'symbol': symbol, 'name': name}
    
    def get_symbol(self, gene_id: str) -> Optional[str]:
        """Get gene symbol from ID"""
        return self.mappings.get(gene_id, {}).get('symbol')
    
    def get_name(self, gene_id: str) -> Optional[str]:
        """Get gene name from ID"""
        return self.mappings.get(gene_id, {}).get('name')
    
    def load_mapping_file(self, filepath: Union[str, Path], 
                         id_column: str = 'gene_id',
                         symbol_column: str = 'gene_symbol',
                         name_column: str = 'gene_name'):
        """Load mappings from a file"""
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            gene_id = row.get(id_column)
            symbol = row.get(symbol_column)
            name = row.get(name_column)
            
            if gene_id:
                self.mappings[gene_id] = {
                    'symbol': symbol,
                    'name': name
                }


class RNASeqImporter:
    """
    Main importer for RNA-seq analysis results
    
    Supports multiple input formats and analysis tools:
    - DESeq2 results files
    - EdgeR results files 
    - Generic CSV/TSV files
    - Expression matrices
    """
    
    def __init__(self, gene_mapper: Optional[GeneIDMapper] = None):
        """
        Initialize RNA-seq importer
        
        Args:
            gene_mapper: Gene ID mapper for identifier conversion
        """
        self.gene_mapper = gene_mapper or GeneIDMapper()
    
    def import_deseq2_results(self, filepath: Union[str, Path], 
                             experiment_id: str,
                             sample_groups: Optional[Dict[str, List[str]]] = None) -> RNASeqExperiment:
        """
        Import DESeq2 results file
        
        Expected columns:
        - gene_id (or row names)
        - baseMean
        - log2FoldChange
        - lfcSE
        - stat
        - pvalue
        - padj
        
        Args:
            filepath: Path to DESeq2 results CSV/TSV file
            experiment_id: Unique experiment identifier
            sample_groups: Sample group definitions
            
        Returns:
            RNASeqExperiment object
        """
        filepath = Path(filepath)
        
        # Read DESeq2 results file
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, index_col=0)
        else:  # Assume TSV
            df = pd.read_csv(filepath, sep='	', index_col=0)
        
        # Map column names (DESeq2 uses specific names)
        column_mapping = {
            'baseMean': 'base_mean',
            'log2FoldChange': 'log2_fold_change', 
            'lfcSE': 'log2_fold_change_se',
            'stat': 'stat',
            'pvalue': 'p_value',
            'padj': 'p_adj'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Create expression data objects
        expression_data = []
        
        for gene_id, row in df.iterrows():
            # Create Gene object with ID mapping
            gene = Gene(
                gene_id=str(gene_id),
                gene_symbol=self.gene_mapper.get_symbol(str(gene_id)),
                gene_name=self.gene_mapper.get_name(str(gene_id))
            )
            
            # Create ExpressionData object
            expr_data = ExpressionData(
                gene=gene,
                base_mean=float(row.get('base_mean', 0)),
                log2_fold_change=float(row.get('log2_fold_change', 0)),
                log2_fold_change_se=row.get('log2_fold_change_se'),
                stat=row.get('stat'),
                p_value=row.get('p_value'),
                p_adj=row.get('p_adj')
            )
            
            expression_data.append(expr_data)
        
        # Create experiment object
        experiment = RNASeqExperiment(
            experiment_id=experiment_id,
            sample_groups=sample_groups or {},
            expression_data=expression_data,
            metadata={
                'source_file': str(filepath),
                'import_date': datetime.now().isoformat(),
                'total_genes': len(expression_data),
                'analysis_tool': 'DESeq2'
            }
        )
        
        return experiment
    
    def import_edger_results(self, filepath: Union[str, Path],
                            experiment_id: str,
                            sample_groups: Optional[Dict[str, List[str]]] = None) -> RNASeqExperiment:
        """
        Import EdgeR results file
        
        Expected columns:
        - gene_id (or row names)
        - logFC
        - logCPM
        - LR (or F)
        - PValue
        - FDR
        
        Args:
            filepath: Path to EdgeR results file
            experiment_id: Unique experiment identifier
            sample_groups: Sample group definitions
            
        Returns:
            RNASeqExperiment object
        """
        filepath = Path(filepath)
        
        # Read EdgeR results file
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, index_col=0)
        else:  # Assume TSV
            df = pd.read_csv(filepath, sep='	', index_col=0)
        
        # Map EdgeR column names to our format
        column_mapping = {
            'logFC': 'log2_fold_change',
            'logCPM': 'base_mean',  # Approximate
            'LR': 'stat',
            'F': 'stat', 
            'PValue': 'p_value',
            'FDR': 'p_adj'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Create expression data objects
        expression_data = []
        
        for gene_id, row in df.iterrows():
            # Create Gene object with ID mapping
            gene = Gene(
                gene_id=str(gene_id),
                gene_symbol=self.gene_mapper.get_symbol(str(gene_id)),
                gene_name=self.gene_mapper.get_name(str(gene_id))
            )
            
            # Create ExpressionData object
            expr_data = ExpressionData(
                gene=gene,
                base_mean=float(row.get('base_mean', 0)),
                log2_fold_change=float(row.get('log2_fold_change', 0)),
                stat=row.get('stat'),
                p_value=row.get('p_value'),
                p_adj=row.get('p_adj')
            )
            
            expression_data.append(expr_data)
        
        # Create experiment object
        experiment = RNASeqExperiment(
            experiment_id=experiment_id,
            sample_groups=sample_groups or {},
            expression_data=expression_data,
            metadata={
                'source_file': str(filepath),
                'import_date': datetime.now().isoformat(),
                'total_genes': len(expression_data),
                'analysis_tool': 'EdgeR'
            }
        )
        
        return experiment
    
    def import_expression_matrix(self, filepath: Union[str, Path],
                                experiment_id: str,
                                sample_groups: Dict[str, List[str]]) -> RNASeqExperiment:
        """
        Import raw expression matrix (genes x samples)
        
        Args:
            filepath: Path to expression matrix CSV/TSV
            experiment_id: Unique experiment identifier
            sample_groups: Sample group definitions
            
        Returns:
            RNASeqExperiment object
        """
        filepath = Path(filepath)
        
        # Read expression matrix
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, index_col=0)
        else:  # Assume TSV
            df = pd.read_csv(filepath, sep='	', index_col=0)
        
        # Create expression data objects (without differential expression stats)
        expression_data = []
        
        for gene_id in df.index:
            # Create Gene object with ID mapping
            gene = Gene(
                gene_id=str(gene_id),
                gene_symbol=self.gene_mapper.get_symbol(str(gene_id)),
                gene_name=self.gene_mapper.get_name(str(gene_id))
            )
            
            # Get counts for all samples
            raw_counts = {col: int(df.loc[gene_id, col]) for col in df.columns}
            
            # Calculate basic statistics
            mean_expr = np.mean(list(raw_counts.values()))
            
            # Create ExpressionData object
            expr_data = ExpressionData(
                gene=gene,
                base_mean=mean_expr,
                log2_fold_change=0.0,  # No comparison performed
                raw_counts=raw_counts
            )
            
            expression_data.append(expr_data)
        
        # Create experiment object
        experiment = RNASeqExperiment(
            experiment_id=experiment_id,
            sample_groups=sample_groups,
            expression_data=expression_data,
            metadata={
                'source_file': str(filepath),
                'import_date': datetime.now().isoformat(),
                'total_genes': len(expression_data),
                'total_samples': len(df.columns),
                'sample_names': list(df.columns),
                'analysis_tool': 'Expression Matrix'
            }
        )
        
        return experiment
    
    def export_experiment(self, experiment: RNASeqExperiment, 
                         output_path: Union[str, Path],
                         format: str = 'csv',
                         include_nonsignificant: bool = True):
        """
        Export experiment data to file
        
        Args:
            experiment: RNASeqExperiment to export
            output_path: Output file path
            format: Export format ('csv', 'tsv', 'json')
            include_nonsignificant: Whether to include non-significant genes
        """
        output_path = Path(output_path)
        
        if format.lower() in ['csv', 'tsv']:
            self._export_tabular(experiment, output_path, format, include_nonsignificant)
        elif format.lower() == 'json':
            self._export_json(experiment, output_path, include_nonsignificant)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_tabular(self, experiment: RNASeqExperiment, 
                       output_path: Path, format: str,
                       include_nonsignificant: bool):
        """Export to CSV/TSV format"""
        sep = ',' if format.lower() == 'csv' else '\t'
        
        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=sep)
            
            # Write header
            header = [
                'gene_id', 'gene_symbol', 'gene_name', 'base_mean',
                'log2_fold_change', 'fold_change', 'p_value', 'p_adj',
                'significant'
            ]
            writer.writerow(header)
            
            # Write data
            for expr in experiment.expression_data:
                if not include_nonsignificant and not expr.is_significant():
                    continue
                
                row = [
                    expr.gene.gene_id,
                    expr.gene.gene_symbol or '',
                    expr.gene.gene_name or '',
                    f"{expr.base_mean:.3f}",
                    f"{expr.log2_fold_change:.3f}",
                    f"{expr.fold_change:.3f}",
                    f"{expr.p_value:.2e}" if expr.p_value else '',
                    f"{expr.p_adj:.2e}" if expr.p_adj else '',
                    str(expr.is_significant())
                ]
                writer.writerow(row)
    
    def _export_json(self, experiment: RNASeqExperiment, 
                    output_path: Path, include_nonsignificant: bool):
        """Export to JSON format"""
        data = {
            'experiment_id': experiment.experiment_id,
            'created_date': experiment.created_date.isoformat(),
            'metadata': experiment.metadata,
            'sample_groups': experiment.sample_groups,
            'summary': {
                'total_genes': len(experiment.expression_data),
                'significant_genes': len(experiment.get_significant_genes()),
                'upregulated': len(experiment.get_upregulated_genes()),
                'downregulated': len(experiment.get_downregulated_genes())
            },
            'genes': []
        }
        
        # Add gene data
        for expr in experiment.expression_data:
            if not include_nonsignificant and not expr.is_significant():
                continue
            
            gene_data = {
                'gene_id': expr.gene.gene_id,
                'gene_symbol': expr.gene.gene_symbol,
                'gene_name': expr.gene.gene_name,
                'base_mean': expr.base_mean,
                'log2_fold_change': expr.log2_fold_change,
                'fold_change': expr.fold_change,
                'p_value': expr.p_value,
                'p_adj': expr.p_adj,
                'significant': expr.is_significant()
            }
            
            if expr.raw_counts:
                gene_data['raw_counts'] = expr.raw_counts
            
            data['genes'].append(gene_data)
        
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=2, default=str)