from typing import Dict, Literal, Optional

import pandas as pd


from .db_dataset import DBDataset

__all__ = ["CTUDataset"]


# fmt: off
CTUDatabaseName = Literal[
    "Accidents", "AdventureWorks2014", "Airline", "Atherosclerosis", "AustralianFootball", 
    "Basketball_men", "Basketball_women", "Biodegradability", "Bupa", "Carcinogenesis", 
    "ccs", "CDESchools", "Chess", "CiteSeer", "classicmodels", "ConsumerExpenditures",
    "CORA", "Countries", "CraftBeer", "Credit", "cs", "Dallas", "DCG", "Dunur", "Elti",
    "employee", "ErgastF1", "Facebook", "financial", "FNHK", "ftp", "geneea", "genes",
    "GOSales", "Grants", "Hepatitis_std", "Hockey", "imdb_ijs", "KRK", "lahman_2014",
    "legalActs", "Mesh", "medical", "Mondial", "Mooney_Family", "imdb_MovieLens",
    "MuskLarge", "MuskSmall", "mutagenesis", "nations", "NBA", "NCAA", "northwind", "Pima",
    "PremierLeague", "PTE", "PubMed_Diabetes", "pubs", "Pyrimidine", "restbase",
    "sakila", "SalesDB", "Same_gen", "SAP", "SAT", "Seznam", "SFScores", "Shakespeare",
    "stats", "Student_loan", "Toxicology", "tpcc", "tpcd", "tpcds", "tpch", "trains",
    "Triazine", "university", "UTube", "UW_std", "VisualGenome", "voc", "Walmart", "WebKP",
    "world"
]
# fmt: on


class CTUDataset(DBDataset):
    # To be set by subclass if available.
    val_timestamp = pd.Timestamp.max.date()
    test_timestamp = pd.Timestamp.max.date()

    def __init__(
        self,
        database: CTUDatabaseName,
        cache_dir: Optional[str] = None,
        time_col_dict: Optional[Dict[str, str]] = None,
        keep_original_keys: bool = False,
        keep_original_compound_keys: bool = True,
    ):
        """Create a database dataset object.

        Args:
            database (CTUDatabaseName): The name of the database.
            cache_dir (str, optional): The directory to cache the dataset. Defaults to None.
            time_col_dict (Dict[str, str], optional): A dictionary mapping table names to time columns. Defaults to None.
            keep_original_keys (bool, optional): Whether to keep original primary and foreign keys \
                after duplication during re-indexing. This is useful when the keys contain information \
                beyond just their relationship to other rows. Defaults to False.
            keep_original_compound_keys (bool, optional): Whether to keep original compound primary \
                and foreign keys as they often contain useful data. Defaults to True.
        """
        self.database = database
        super().__init__(
            cache_dir=cache_dir,
            dialect="mariadb",
            driver="pymysql",
            user="guest",
            password="ctu-relational",
            host="relational.fel.cvut.cz",
            port=3306,
            database=database,
            time_col_dict=time_col_dict,
            keep_original_keys=keep_original_keys,
            keep_original_compound_keys=keep_original_compound_keys,
        )
