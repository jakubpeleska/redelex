import os

import pandas as pd

from relbench.datasets import register_dataset

from .db_dataset import DBDataset
from .ctu_dataset import CTUDataset

# fmt: off
from .ctu_datasets import (
    Accidents, AdventureWorks, Airline, Atherosclerosis, BasketballMen,
    BasketballWomen, Biodegradability, Bupa, Carcinogenesis,
    CDESchools, Chess, ClassicModels, CORA, Countries, CraftBeer, Credit,
    Dallas, DCG, Diabetes, Dunur, Elti, ErgastF1,
    Expenditures, Employee, Financial, FNHK, FTP, Geneea, Genes, GOSales,
    Grants, Hepatitis, Hockey, IMDb, Lahman, LegalActs, Mesh,
    Mondial, Mooney, MovieLens, MuskLarge, MuskSmall, Mutagenesis,
    NCAA, Northwind, Pima, PremiereLeague, Restbase, Sakila,
    Sales, SameGen, SAP, Satellite, Seznam, SFScores, Shakespeare, Stats,
    StudentLoan, Thrombosis, Toxicology, TPCC, TPCD, TPCDS, TPCH, Triazine,
    UWCSE, VisualGenome, VOC, Walmart, WebKP, World
)
# fmt: on


def get_dataset_info(dataset_name: str):
    info_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset-info.csv"))
    return info_df[info_df["dataset"] == dataset_name].iloc[0]


register_dataset("ctu-accidents", Accidents)
register_dataset("ctu-adventureworks", AdventureWorks)
register_dataset("ctu-airline", Airline)
register_dataset("ctu-atherosclerosis", Atherosclerosis)
register_dataset("ctu-basketballmen", BasketballMen)
register_dataset("ctu-basketballwomen", BasketballWomen)
register_dataset("ctu-biodegradability", Biodegradability)
register_dataset("ctu-bupa", Bupa)
register_dataset("ctu-carcinogenesis", Carcinogenesis)
register_dataset("ctu-cde", CDESchools)
register_dataset("ctu-chess", Chess)
register_dataset("ctu-classicmodels", ClassicModels)
register_dataset("ctu-cora", CORA)
register_dataset("ctu-countries", Countries)
register_dataset("ctu-craftbeer", CraftBeer)
register_dataset("ctu-credit", Credit)
register_dataset("ctu-dallas", Dallas)
register_dataset("ctu-dcg", DCG)
register_dataset("ctu-diabetes", Diabetes)
register_dataset("ctu-dunur", Dunur)
register_dataset("ctu-elti", Elti)
register_dataset("ctu-employee", Employee)
register_dataset("ctu-ergastf1", ErgastF1)
register_dataset("ctu-expenditures", Expenditures)
register_dataset("ctu-employee", Employee)
register_dataset("ctu-financial", Financial)
register_dataset("ctu-fnhk", FNHK)
register_dataset("ctu-ftp", FTP)
register_dataset("ctu-geneea", Geneea)
register_dataset("ctu-genes", Genes)
register_dataset("ctu-gosales", GOSales)
register_dataset("ctu-grants", Grants)
register_dataset("ctu-hepatitis", Hepatitis)
register_dataset("ctu-hockey", Hockey)
register_dataset("ctu-imdb", IMDb)
register_dataset("ctu-lahman", Lahman)
register_dataset("ctu-legalacts", LegalActs)
register_dataset("ctu-mesh", Mesh)
register_dataset("ctu-mondial", Mondial)
register_dataset("ctu-mooney", Mooney)
register_dataset("ctu-movielens", MovieLens)
register_dataset("ctu-musklarge", MuskLarge)
register_dataset("ctu-musksmall", MuskSmall)
register_dataset("ctu-mutagenesis", Mutagenesis)
register_dataset("ctu-ncaa", NCAA)
register_dataset("ctu-northwind", Northwind)
register_dataset("ctu-pima", Pima)
register_dataset("ctu-premiereleague", PremiereLeague)
register_dataset("ctu-restbase", Restbase)
register_dataset("ctu-sakila", Sakila)
register_dataset("ctu-sales", Sales)
register_dataset("ctu-samegen", SameGen)
register_dataset("ctu-sap", SAP)
register_dataset("ctu-satellite", Satellite)
register_dataset("ctu-seznam", Seznam)
register_dataset("ctu-sfscores", SFScores)
register_dataset("ctu-shakespeare", Shakespeare)
register_dataset("ctu-stats", Stats)
register_dataset("ctu-studentloan", StudentLoan)
register_dataset("ctu-thrombosis", Thrombosis)
register_dataset("ctu-toxicology", Toxicology)
register_dataset("ctu-tpcc", TPCC)
register_dataset("ctu-tpcd", TPCD)
register_dataset("ctu-tpcds", TPCDS)
register_dataset("ctu-tpch", TPCH)
register_dataset("ctu-triazine", Triazine)
register_dataset("ctu-uwcse", UWCSE)
register_dataset("ctu-visualgenome", VisualGenome)
register_dataset("ctu-voc", VOC)
register_dataset("ctu-walmart", Walmart)
register_dataset("ctu-webkp", WebKP)
register_dataset("ctu-world", World)

# fmt: off
__all__ = [
    "get_dataset_info",
    
    "DBDataset", "CTUDataset",
    
    "Accidents", "AdventureWorks", "Airline", "Atherosclerosis", "BasketballMen",
    "BasketballWomen", "Biodegradability", "Bupa", "Carcinogenesis",
    "CDESchools", "Chess", "ClassicModels", "CORA", "Countries", "CraftBeer", "Credit",
    "Dallas", "DCG", "Diabetes", "Dunur", "Elti", "Employee", "ErgastF1",
    "Expenditures", "Employee", "Financial", "FNHK", "FTP", "Geneea", "Genes", "GOSales",
    "Grants", "Hepatitis", "Hockey", "IMDb", "Lahman", "LegalActs", "Mesh",
    "Mondial", "Mooney", "MovieLens", "MuskLarge", "MuskSmall", "Mutagenesis",
    "NCAA", "Northwind", "Pima", "PremiereLeague", "Restbase", "Sakila",
    "Sales", "SameGen", "SAP", "Satellite", "Seznam", "SFScores", "Shakespeare", "Stats",
    "StudentLoan", "Thrombosis", "Toxicology", "TPCC", "TPCD", "TPCDS", "TPCH", "Triazine",
    "UWCSE", "VisualGenome", "VOC", "Walmart", "WebKP", "World"
]
# fmt: on
