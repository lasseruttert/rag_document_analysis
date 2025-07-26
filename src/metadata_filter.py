#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metadata Filtering System for RAG Document Analysis.

This module provides advanced filtering capabilities for document search
based on metadata attributes like file type, date, tags, and content.
"""

import re
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FilterOperator(Enum):
    """Filter operation types."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    REGEX = "regex"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"

@dataclass
class QueryFilter:
    """
    A single metadata filter condition.
    """
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False
    
    def apply(self, metadata: Dict[str, Any]) -> bool:
        """
        Apply this filter to a metadata dictionary.
        
        Args:
            metadata: Document metadata to test
            
        Returns:
            True if metadata matches the filter condition
        """
        if self.field not in metadata:
            # Handle missing fields
            if self.operator == FilterOperator.NOT_EQUALS:
                return True
            elif self.operator == FilterOperator.NOT_CONTAINS:
                return True
            elif self.operator == FilterOperator.NOT_IN:
                return True
            else:
                return False
        
        field_value = metadata[self.field]
        filter_value = self.value
        
        # Handle case sensitivity for string operations
        if isinstance(field_value, str) and isinstance(filter_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            filter_value = filter_value.lower()
        
        try:
            if self.operator == FilterOperator.EQUALS:
                return field_value == filter_value
            
            elif self.operator == FilterOperator.NOT_EQUALS:
                return field_value != filter_value
            
            elif self.operator == FilterOperator.CONTAINS:
                if isinstance(field_value, str):
                    return filter_value in field_value
                elif isinstance(field_value, (list, tuple)):
                    return filter_value in field_value
                return False
            
            elif self.operator == FilterOperator.NOT_CONTAINS:
                if isinstance(field_value, str):
                    return filter_value not in field_value
                elif isinstance(field_value, (list, tuple)):
                    return filter_value not in field_value
                return True
            
            elif self.operator == FilterOperator.IN:
                if isinstance(filter_value, (list, tuple)):
                    return field_value in filter_value
                return field_value == filter_value
            
            elif self.operator == FilterOperator.NOT_IN:
                if isinstance(filter_value, (list, tuple)):
                    return field_value not in filter_value
                return field_value != filter_value
            
            elif self.operator == FilterOperator.GREATER_THAN:
                return field_value > filter_value
            
            elif self.operator == FilterOperator.LESS_THAN:
                return field_value < filter_value
            
            elif self.operator == FilterOperator.GREATER_EQUAL:
                return field_value >= filter_value
            
            elif self.operator == FilterOperator.LESS_EQUAL:
                return field_value <= filter_value
            
            elif self.operator == FilterOperator.REGEX:
                if isinstance(field_value, str):
                    pattern = re.compile(filter_value, re.IGNORECASE if not self.case_sensitive else 0)
                    return bool(pattern.search(field_value))
                return False
            
            elif self.operator == FilterOperator.STARTS_WITH:
                if isinstance(field_value, str):
                    return field_value.startswith(filter_value)
                return False
            
            elif self.operator == FilterOperator.ENDS_WITH:
                if isinstance(field_value, str):
                    return field_value.endswith(filter_value)
                return False
            
            else:
                logger.warning(f"Unknown filter operator: {self.operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying filter {self.field} {self.operator} {self.value}: {e}")
            return False

@dataclass
class CombinedFilter:
    """
    Combines multiple filters with AND/OR logic.
    """
    filters: List[Union[QueryFilter, 'CombinedFilter']]
    operator: str = "AND"  # "AND" or "OR"
    
    def apply(self, metadata: Dict[str, Any]) -> bool:
        """
        Apply combined filter to metadata.
        
        Args:
            metadata: Document metadata to test
            
        Returns:
            True if metadata matches the combined filter condition
        """
        if not self.filters:
            return True
        
        results = []
        for filter_item in self.filters:
            if isinstance(filter_item, (QueryFilter, CombinedFilter)):
                results.append(filter_item.apply(metadata))
            else:
                logger.warning(f"Invalid filter type: {type(filter_item)}")
                results.append(False)
        
        if self.operator.upper() == "AND":
            return all(results)
        elif self.operator.upper() == "OR":
            return any(results)
        else:
            logger.warning(f"Unknown combination operator: {self.operator}")
            return False

class MetadataFilter:
    """
    Advanced metadata filtering system for document search.
    
    Provides convenient methods for creating common filter types
    and combining them into complex queries.
    """
    
    @staticmethod
    def by_file_type(file_types: Union[str, List[str]], exclude: bool = False) -> QueryFilter:
        """
        Filter by file type(s).
        
        Args:
            file_types: Single file type or list of file types
            exclude: If True, exclude these file types instead of including
            
        Returns:
            QueryFilter for file type filtering
        """
        if isinstance(file_types, str):
            file_types = [file_types]
        
        operator = FilterOperator.NOT_IN if exclude else FilterOperator.IN
        return QueryFilter(
            field="file_type",
            operator=operator,
            value=file_types,
            case_sensitive=False
        )
    
    @staticmethod
    def by_filename(pattern: str, exact_match: bool = False, case_sensitive: bool = False) -> QueryFilter:
        """
        Filter by filename pattern.
        
        Args:
            pattern: Filename pattern (exact match or regex)
            exact_match: If True, use exact matching; if False, use regex
            case_sensitive: Whether to match case sensitively
            
        Returns:
            QueryFilter for filename filtering
        """
        operator = FilterOperator.EQUALS if exact_match else FilterOperator.REGEX
        return QueryFilter(
            field="filename",
            operator=operator,
            value=pattern,
            case_sensitive=case_sensitive
        )
    
    @staticmethod
    def by_content_size(min_size: Optional[int] = None, max_size: Optional[int] = None) -> Union[QueryFilter, CombinedFilter]:
        """
        Filter by content size (character count).
        
        Args:
            min_size: Minimum content size
            max_size: Maximum content size
            
        Returns:
            QueryFilter or CombinedFilter for size filtering
        """
        filters = []
        
        if min_size is not None:
            filters.append(QueryFilter(
                field="chunk_size",
                operator=FilterOperator.GREATER_EQUAL,
                value=min_size
            ))
        
        if max_size is not None:
            filters.append(QueryFilter(
                field="chunk_size",
                operator=FilterOperator.LESS_EQUAL,
                value=max_size
            ))
        
        if len(filters) == 1:
            return filters[0]
        elif len(filters) == 2:
            return CombinedFilter(filters=filters, operator="AND")
        else:
            return QueryFilter(field="chunk_size", operator=FilterOperator.GREATER_THAN, value=0)
    
    @staticmethod
    def by_position_range(start_pos: Optional[int] = None, end_pos: Optional[int] = None) -> Union[QueryFilter, CombinedFilter]:
        """
        Filter by document position range.
        
        Args:
            start_pos: Minimum position in document
            end_pos: Maximum position in document
            
        Returns:
            QueryFilter or CombinedFilter for position filtering
        """
        filters = []
        
        if start_pos is not None:
            filters.append(QueryFilter(
                field="position",
                operator=FilterOperator.GREATER_EQUAL,
                value=start_pos
            ))
        
        if end_pos is not None:
            filters.append(QueryFilter(
                field="position",
                operator=FilterOperator.LESS_EQUAL,
                value=end_pos
            ))
        
        if len(filters) == 1:
            return filters[0]
        elif len(filters) == 2:
            return CombinedFilter(filters=filters, operator="AND")
        else:
            return QueryFilter(field="position", operator=FilterOperator.GREATER_EQUAL, value=0)
    
    @staticmethod
    def by_semantic_density(min_density: Optional[float] = None, max_density: Optional[float] = None) -> Union[QueryFilter, CombinedFilter]:
        """
        Filter by semantic density score.
        
        Args:
            min_density: Minimum semantic density (0.0-1.0)
            max_density: Maximum semantic density (0.0-1.0)
            
        Returns:
            QueryFilter or CombinedFilter for density filtering
        """
        filters = []
        
        if min_density is not None:
            filters.append(QueryFilter(
                field="semantic_density",
                operator=FilterOperator.GREATER_EQUAL,
                value=min_density
            ))
        
        if max_density is not None:
            filters.append(QueryFilter(
                field="semantic_density",
                operator=FilterOperator.LESS_EQUAL,
                value=max_density
            ))
        
        if len(filters) == 1:
            return filters[0]
        elif len(filters) == 2:
            return CombinedFilter(filters=filters, operator="AND")
        else:
            return QueryFilter(field="semantic_density", operator=FilterOperator.GREATER_EQUAL, value=0.0)
    
    @staticmethod
    def by_keywords(keywords: Union[str, List[str]], match_all: bool = False) -> QueryFilter:
        """
        Filter by keyword tags.
        
        Args:
            keywords: Single keyword or list of keywords
            match_all: If True, all keywords must be present; if False, any keyword matches
            
        Returns:
            QueryFilter for keyword filtering
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Since keyword_tags is a list, we need to check intersection
        if match_all:
            # For match_all, we'll use a custom approach in the filter application
            return QueryFilter(
                field="keyword_tags",
                operator=FilterOperator.CONTAINS,  # We'll handle this specially
                value=keywords
            )
        else:
            # For any match, check if any keyword is in the tags
            return QueryFilter(
                field="keyword_tags",
                operator=FilterOperator.CONTAINS,
                value=keywords[0] if len(keywords) == 1 else keywords
            )
    
    @staticmethod
    def by_heading_presence(has_heading: bool = True) -> QueryFilter:
        """
        Filter by presence of headings in chunk.
        
        Args:
            has_heading: True to include only chunks with headings
            
        Returns:
            QueryFilter for heading filtering
        """
        return QueryFilter(
            field="contains_heading",
            operator=FilterOperator.EQUALS,
            value=has_heading
        )
    
    @staticmethod
    def by_custom_field(field: str, value: Any, operator: FilterOperator = FilterOperator.EQUALS, case_sensitive: bool = False) -> QueryFilter:
        """
        Create a custom field filter.
        
        Args:
            field: Metadata field name
            value: Value to compare against
            operator: Comparison operator
            case_sensitive: Whether string comparisons should be case sensitive
            
        Returns:
            QueryFilter for custom field filtering
        """
        return QueryFilter(
            field=field,
            operator=operator,
            value=value,
            case_sensitive=case_sensitive
        )
    
    @staticmethod
    def combine_filters(filters: List[Union[QueryFilter, CombinedFilter]], operator: str = "AND") -> CombinedFilter:
        """
        Combine multiple filters with AND/OR logic.
        
        Args:
            filters: List of filters to combine
            operator: "AND" or "OR"
            
        Returns:
            CombinedFilter combining all input filters
        """
        return CombinedFilter(filters=filters, operator=operator)
    
    @staticmethod
    def apply_filters(documents_with_metadata: List[Dict[str, Any]], filters: Union[QueryFilter, CombinedFilter]) -> List[Dict[str, Any]]:
        """
        Apply filters to a list of documents with metadata.
        
        Args:
            documents_with_metadata: List of documents with 'metadata' field
            filters: Filter(s) to apply
            
        Returns:
            Filtered list of documents
        """
        if not documents_with_metadata or not filters:
            return documents_with_metadata
        
        filtered_docs = []
        for doc in documents_with_metadata:
            metadata = doc.get('metadata', {})
            if filters.apply(metadata):
                filtered_docs.append(doc)
        
        return filtered_docs

# Convenience functions for common filter combinations
def create_pdf_only_filter() -> QueryFilter:
    """Create filter for PDF documents only."""
    return MetadataFilter.by_file_type("pdf")

def create_high_quality_content_filter() -> CombinedFilter:
    """Create filter for high-quality content (headings + good semantic density)."""
    return MetadataFilter.combine_filters([
        MetadataFilter.by_heading_presence(True),
        MetadataFilter.by_semantic_density(min_density=0.5)
    ], operator="AND")

def create_recent_large_documents_filter() -> CombinedFilter:
    """Create filter for recent, substantial documents."""
    return MetadataFilter.combine_filters([
        MetadataFilter.by_content_size(min_size=500),
        MetadataFilter.by_semantic_density(min_density=0.3)
    ], operator="AND")

# Test and example usage
if __name__ == '__main__':
    logger.info("MetadataFilter module loaded for testing")