from typing import Optional

import numpy as np
import pandas as pd

from relbench.base import Database, Table

from redelex.utils import TIMESTAMP_MIN, TIMESTAMP_MAX

from .ctu_dataset import CTUDataset

# fmt: off
__all__ = [    
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


class Accidents(CTUDataset):
    """
    Traffic accident database consists of all accidents that happened in Slovenia's\
    capital city Ljubljana between the years 1995 and 2006.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Accidents",
            cache_dir=cache_dir,
            time_col_dict={"nesreca": "cas_nesreca"},
            keep_original_keys=False,
        )


class AdventureWorks(CTUDataset):
    """
    Adventure Works 2014 (OLTP version) is a sample database for Microsoft SQL Server, \
    which has replaced Northwind and Pub sample databases that were shipped earlier. \
    The database is about a fictious, multinational bicycle manufacturer called \
    Adventure Works Cycles.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "AdventureWorks2014",
            cache_dir=cache_dir,
            time_col_dict={
                # "BillOfMaterials": "StartDate",
                # "Employee": "HireDate",
                # "EmployeePayHistory": "RateChangeDate",
                # "CurrencyRate": "CurrencyRateDate",
                # "Product": "SellStartDate",
                # "ProductCostHistory": "StartDate",
                # "ProductListPriceHistory": "StartDate",
                # "ProductReview": "ReviewDate",
                # "PurchaseOrderHeader": "OrderDate",
                "SalesOrderHeader": "OrderDate",
                # "SalesPersonQuotaHistory": "QuotaDate",
                # "SalesTerritoryHistory": "StartDate",
                # "SpecialOffer": "StartDate",
                # "TransactionHistory": "TransactionDate",
                # "WorkOrder": "StartDate",
                # "WorkOrderRouting": "ScheduledStartDate",
                # "EmployeeDepartmentHistory": "StartDate",
            },
            keep_original_keys=False,
            keep_original_compound_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict.pop("AWBuildVersion", None)
        db.table_dict.pop("DatabaseLog", None)
        db.table_dict.pop("ErrorLog", None)
        db.table_dict.pop("TransactionHistoryArchive", None)

        db.table_dict["Address"].df.drop(columns=["SpatialLocation"], inplace=True)
        db.table_dict["Document"].df.drop(columns=["Document"], inplace=True)
        db.table_dict["ProductPhoto"].df.drop(columns=["ThumbNailPhoto"], inplace=True)
        db.table_dict["ProductPhoto"].df.drop(columns=["LargePhoto"], inplace=True)

        return db


class Airline(CTUDataset):
    """
    Airline on-time data are reported each month to the U.S. Department of Transportation\
    (DOT), Bureau of Transportation Statistics (BTS) by the 16 U.S. air carriers that have\
    at least 1 percent of total domestic scheduled-service passenger revenues, plus two\
    other carriers that report voluntarily. The data cover nonstop scheduled-service\
    flights between points within the United States (including territories) as described\
    in 14 CFR Part 234 of DOT's regulations. Data are available since January 1995.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Airline",
            cache_dir=cache_dir,
            time_col_dict={"On_Time_On_Time_Performance_2016_1": "FlightDate"},
            keep_original_keys=False,
        )


class Atherosclerosis(CTUDataset):
    """
    The study STULONG is a longitudinal 20 years lasting primary preventive study of\
    middle-aged men. The study aims to identify prevalence of atherosclerosis RFs in\
    a population generally considered to be the most endangered by possible atherosclerosis\
    complications, i.e., middle-aged men.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Atherosclerosis",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class BasketballMen(CTUDataset):
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Basketball_men",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["players"].df["birthDate"] = pd.to_datetime(
            db.table_dict["players"].df["birthDate"], errors="coerce"
        )
        db.table_dict["players"].df["deathDate"] = pd.to_datetime(
            db.table_dict["players"].df["deathDate"], errors="coerce"
        )

        return db


class BasketballWomen(CTUDataset):
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Basketball_women",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Biodegradability(CTUDataset):
    """
    This is an older data set of chemical structures containing 328 compounds\
    labeled by their half-life for aerobic aqueous biodegradation.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Biodegradability",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Bupa(CTUDataset):
    """Evaluation of patients on liver disorder."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Bupa",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class Carcinogenesis(CTUDataset):
    """
    For prediction of whether a given molecule is carcinogenic or not.\
    The dataset contains 182 positive carcinogenicity tests and 148 negative tests.
    """

    ""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Carcinogenesis",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class CDESchools(CTUDataset):
    """
    A database containing geospatial information, as well as SAT average scores and\
    Free-or-Reduced-Price Meal eligibility data, for California schools.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "CDESchools",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Chess(CTUDataset):
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Chess",
            cache_dir=cache_dir,
            time_col_dict={"game": "event_date"},
            keep_original_keys=False,
        )


class ClassicModels(CTUDataset):
    """
    The schema is for Classic Models, a retailer of scale models of classic cars.\
    The database contains typical business data such as customers, orders, order\
    line items, products and so on.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "classicmodels",
            cache_dir=cache_dir,
            time_col_dict={"orders": "orderDate", "payments": "paymentDate"},
            keep_original_keys=False,
        )


class CORA(CTUDataset):
    """
    The Cora dataset consists of 2708 scientific publications classified into one of\
    seven classes. The citation network consists of 5429 links. Each publication in the\
    dataset is described by a 0/1-valued word vector indicating the absence/presence of\
    the corresponding word from the dictionary. The dictionary consists of 1433 unique\
    words.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "CORA",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Countries(CTUDataset):
    """
    Data of forest area for 247 countries.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Countries",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class CraftBeer(CTUDataset):
    """
    Craft beers labeled by styles and composition.\
    A separate dataset lists breweries by state.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "CraftBeer",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Credit(CTUDataset):
    """
    A bit more complex artificial database with loops.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Credit",
            cache_dir=cache_dir,
            time_col_dict={
                "charge": "charge_dt",
                "payment": "payment_dt",
            },
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        # TODO: Keep images and use them as features
        db.table_dict["member"].df.drop(columns=["photograph"], inplace=True)

        return db


class Dallas(CTUDataset):
    """
    Officer-involved shootings as disclosed by the Dallas Police Department.\
    Includes separate tables for officer and subject/suspect information.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Dallas",
            cache_dir=cache_dir,
            time_col_dict={"incidents": "date"},
            keep_original_keys=False,
        )


class DCG(CTUDataset):
    """
    The set of positive examples consists of all sentences of up to seven words that can\
    be generated by the DCG in Bratko's book (565 positive examples). The set of negative\
    examples was generated by randomly selecting one word in each positive example and\
    replacing it by a randomly selected word the leads to an incorrect sentence, according\
    to the grammar (565 negative examples).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "DCG",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Diabetes(CTUDataset):
    """
    The Diabetes dataset consists of 19717 scientific publications from PubMed database\
    pertaining to diabetes classified into one of three classes. The citation network\
    consists of 44338 links. Each publication in the dataset is described by a TF/IDF\
    weighted word vector from a dictionary which consists of 500 unique words.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "PubMed_Diabetes",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Dunur(CTUDataset):
    """
    Dunur is a relatedness of two people due to marriage such that A is dunur of B if\
    a child of A is married to a child of B.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Dunur",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Elti(CTUDataset):
    """
    Elti is a relatedness of two people due to marriage such that A is elti of B if\
    A's husband is a brother of B's husband.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Elti",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Employee(CTUDataset):
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "employee",
            cache_dir=cache_dir,
            time_col_dict={
                "dept_emp": "from_date",
                "dept_manager": "from_date",
                "employees": "hire_date",
                "salaries": "from_date",
                "titles": "from_date",
            },
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        # Drop redundant key columns
        db.table_dict["titles"].df.drop(columns=["emp_no"], inplace=True)
        db.table_dict["dept_manager"].df.drop(columns=["emp_no", "dept_no"], inplace=True)
        db.table_dict["departments"].df.drop(columns=["dept_no"], inplace=True)
        db.table_dict["salaries"].df.drop(columns=["emp_no"], inplace=True)
        db.table_dict["dept_emp"].df.drop(columns=["emp_no", "dept_no"], inplace=True)
        db.table_dict["employees"].df.drop(columns=["emp_no"], inplace=True)

        # Convert date columns to datetime and handle NaT values
        db.table_dict["titles"].df["to_date"] = pd.to_datetime(
            db.table_dict["titles"].df["to_date"], errors="coerce"
        )
        db.table_dict["dept_manager"].df["to_date"] = pd.to_datetime(
            db.table_dict["dept_manager"].df["to_date"], errors="coerce"
        )
        db.table_dict["salaries"].df["to_date"] = pd.to_datetime(
            db.table_dict["salaries"].df["to_date"], errors="coerce"
        )
        db.table_dict["dept_emp"].df["to_date"] = pd.to_datetime(
            db.table_dict["dept_emp"].df["to_date"], errors="coerce"
        )

        for table in db.table_dict.values():
            dtcols = table.df.select_dtypes(include=["datetime"])
            dtcols[dtcols > pd.Timestamp("2100-01-01")] = pd.NaT
            dtcols[dtcols < pd.Timestamp("1900-01-01")] = pd.NaT
            table.df[dtcols.columns] = dtcols

        return db


class ErgastF1(CTUDataset):
    """
    Ergast.com is a webservice that provides a database of Formula 1 races, \
    starting from the 1950 season until today. The dataset includes information \
    such as the time taken in each lap, the time taken for pit stops, the performance \
    in the qualifying rounds etc. of all Formula 1 races from 1950 to 2017.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "ErgastF1",
            cache_dir=cache_dir,
            time_col_dict={"races": "date"},
            keep_original_keys=False,
            keep_original_compound_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        # Convert time column to datetime
        db.table_dict["pitStops"].df["time"] = (
            db.table_dict["pitStops"].df["time"].fillna(pd.Timedelta(hours=12))
        )
        db.table_dict["pitStops"].df["time"] += db.table_dict["pitStops"].df.join(
            db.table_dict["races"].df["date"], on="FK_races_raceId", how="left"
        )["date"]
        db.table_dict["pitStops"].time_col = "time"

        # Merge date and time columns
        db.table_dict["races"].df["time"] = (
            db.table_dict["races"].df["time"].fillna(pd.Timedelta(hours=12))
        )
        db.table_dict["races"].df["date"] += db.table_dict["races"].df["time"]
        db.table_dict["races"].df.drop(columns=["time"], inplace=True)

        # Convert time columns to datetime
        db.table_dict["qualifying"].df["q1"] = pd.to_datetime(
            db.table_dict["qualifying"].df["q1"], format="%M:%S.%f", errors="coerce"
        )
        db.table_dict["qualifying"].df["q2"] = pd.to_datetime(
            db.table_dict["qualifying"].df["q2"], format="%M:%S.%f", errors="coerce"
        )
        db.table_dict["qualifying"].df["q3"] = pd.to_datetime(
            db.table_dict["qualifying"].df["q3"], format="%M:%S.%f", errors="coerce"
        )
        db.table_dict["results"].df["fastestLapTime"] = pd.to_datetime(
            db.table_dict["results"].df["fastestLapTime"],
            format="%M:%S.%f",
            errors="coerce",
        )

        db.table_dict["target"].df["date"] = db.table_dict["target"].df.join(
            db.table_dict["races"].df.set_index("__PK__")["date"], on="raceId", how="left"
        )["date"]
        db.table_dict["target"].time_col = "date"
        db.table_dict["target"].df.drop(columns=["raceId", "driverId"], inplace=True)

        return db


class Expenditures(CTUDataset):
    """
    The Consumer Expenditure Survey (CE) collects data on expenditures, income, \
    and demographics in the United States. The public-use microdata (PUMD) files \
    provide this information for individual respondents without any information that \
    could identify respondents. PUMD files include adjustments for information that is \
    missing because respondents were unwilling or unable to provide it. The files also \
    have been adjusted to reduce the likelihood of identifying respondents, either \
    directly or through inference. The task is to predict, whether the expenditure
    is a gift or not. Household ids change from year to year - this is a property of the \
    data source.
    
    Original source: www.bls.gov
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "ConsumerExpenditures",
            cache_dir=cache_dir,
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        # Remove remove split column
        db.table_dict["EXPENDITURES"].df.drop(columns=["IS_TRAINING"], inplace=True)

        return db


class Financial(CTUDataset):
    """
    PKDD'99 Financial dataset contains 606 successful and 76 not \
    successful loans along with their information and transactions. 
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "financial",
            cache_dir=cache_dir,
            time_col_dict={
                "account": "date",
                "card": "issued",
                "loan": "date",
                "trans": "date",
            },
            keep_original_keys=False,
        )


class FNHK(CTUDataset):
    """
    Anonymised data from a hospital in Hradec Kralove, Czech Republic, \
    about treatment and medication.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "FNHK",
            cache_dir=cache_dir,
            time_col_dict={
                "pripady": "Datum_prijeti",
                "vykony": "Datum_provedeni_vykonu",
                "zup": "Datum_provedeni_vykonu",
            },
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        # Drop redundant key columns
        db.table_dict["pripady"].df.drop(columns=["Identifikace_pripadu"], inplace=True)
        db.table_dict["vykony"].df.drop(columns=["Identifikace_pripadu"], inplace=True)
        db.table_dict["zup"].df.drop(columns=["Identifikace_pripadu"], inplace=True)

        return db


class FTP(CTUDataset):
    """
    PAKDD'15 Data Mining Competition: The data were obtained from simulations of product\
    viewing activities of users with known gender. The data closely follow the real-life\
    distribution in that regard.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "ftp",
            cache_dir=cache_dir,
            time_col_dict={"session": "start_time"},
            keep_original_keys=False,
        )


class Geneea(CTUDataset):
    """
    Data on deputies and senators in the Czech Republic.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "geneea",
            cache_dir=cache_dir,
            time_col_dict={"hl_hlasovani": "datum", "omluvy": "den"},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        # Combine date and time columns
        db.table_dict["hl_hlasovani"].df["cas"] = (
            db.table_dict["hl_hlasovani"].df["datum"]
            + db.table_dict["hl_hlasovani"].df["cas"]
        )

        db.table_dict["omluvy"].df["den"] = pd.to_datetime(
            db.table_dict["omluvy"].df["den"]
        )

        return db


class Genes(CTUDataset):
    """
    KDD Cup 2001 prediction of gene/protein function and localization.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "genes",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class GOSales(CTUDataset):
    """
    GO Sales dataset from IBM contains information about daily sales, methods, retailers,\
    and products of a fictitious outdoor equipment retail chain “Great Outdoors” (GO).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "GOSales",
            cache_dir=cache_dir,
            time_col_dict={"go_daily_sales": "Date"},
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        fk_col, fk_name = self._reindex_fk(
            {name: t.df for name, t in db.table_dict.items()},
            "go_daily_sales",
            ["Product number"],
            "go_products",
            ["Product number"],
        )
        db.table_dict["go_daily_sales"].df[fk_name] = fk_col
        db.table_dict["go_daily_sales"].fkey_col_to_pkey_table[fk_name] = "go_products"

        fk_col, fk_name = self._reindex_fk(
            {name: t.df for name, t in db.table_dict.items()},
            "go_daily_sales",
            ["Retailer code"],
            "go_retailers",
            ["Retailer code"],
        )
        db.table_dict["go_daily_sales"].df[fk_name] = fk_col
        db.table_dict["go_daily_sales"].fkey_col_to_pkey_table[fk_name] = "go_retailers"

        db.table_dict["go_products"].df.drop(columns=["Product number"], inplace=True)
        db.table_dict["go_retailers"].df.drop(columns=["Retailer code"], inplace=True)
        db.table_dict["go_daily_sales"].df.drop(
            columns=["Retailer code", "Product number"], inplace=True
        )

        db.table_dict.pop("go_1k")

        return db


class Grants(CTUDataset):
    """
    This dataset includes funding grants from the National Science Foundation.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Grants",
            cache_dir=cache_dir,
            time_col_dict={
                "awards": "award_effective_date",
                "investigator_awards": "start_date",
            },
            keep_original_keys=False,
        )


class Hepatitis(CTUDataset):
    """
    PKDD'02 Hepatitis dataset describes 206 instances of Hepatitis B \
    (contrasting them against 484 cases of Hepatitis C).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Hepatitis_std",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Hockey(CTUDataset):
    """
    The Hockey Database, in addition to the NHL, covers the following early and\
    alternative leagues: NHA, PCHA, WCHL and WHA. It contains individual and team\
    statistics from 1909-10 through the 2011-12 season.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Hockey",
            cache_dir=cache_dir,
            time_col_dict={
                "CombinedShutouts": "date",
            },
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["CombinedShutouts"].df = db.table_dict["CombinedShutouts"].df.rename(
            columns={"date": "day"}
        )
        db.table_dict["CombinedShutouts"].df["date"] = pd.to_datetime(
            db.table_dict["CombinedShutouts"].df[["year", "month", "day"]]
        )
        db.table_dict["CombinedShutouts"].df.drop(
            columns=["year", "month", "day"], inplace=True
        )

        for table in db.table_dict.values():
            if "year" in table.df.columns:
                table.df["year"] = pd.to_datetime(table.df["year"], format="%Y")
                table.time_col = "year"

        # Remove tables with no foreign keys
        db.table_dict.pop("abbrev")
        db.table_dict.pop("AwardsMisc")
        # TODO: Possibly keep the table and fix FK to Master
        db.table_dict.pop("HOF")

        return db


class IMDb(CTUDataset):
    """
    The IMDb database: moderately large, real database of movies.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "imdb_ijs",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["movies"].df["year"] = pd.to_datetime(
            db.table_dict["movies"].df["year"], format="%Y"
        )
        db.table_dict["movies"].time_col = "year"

        return db


class Lahman(CTUDataset):
    """
    Lahman's baseball database contains complete batting and pitching statistics\
    from 1871 to 2014, plus fielding statistics, standings, team stats, managerial\
    records, post-season data, and more.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "lahman_2014",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        for table in db.table_dict.values():
            if "yearID" in table.df.columns:
                table.df["yearID"] = pd.to_datetime(
                    table.df["yearID"], format="%Y", errors="coerce"
                )
                table.time_col = "yearID"

        return db


class LegalActs(CTUDataset):
    """
    Bulgarian court decision metadata.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "legalActs",
            cache_dir=cache_dir,
            time_col_dict={
                "legalacts": "StartDate",
            },
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        # Remove scrape fix table
        db.table_dict.pop("scrapefix")

        return db


class Mesh(CTUDataset):
    """
    This domain is about finite element methods in engineering where the number of\
    elements in the mesh model can vary between 1 and 17.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Mesh",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class Mondial(CTUDataset):
    """
    A geography dataset from University of Göttingen describes 114 Christian countries\
    and 71 non-Christian countries.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Mondial",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["organization"].df["Established"] = (
            db.table_dict["organization"].df["Established"].dt.year
        )
        db.table_dict["politics"].df["Independence"] = (
            db.table_dict["politics"].df["Independence"].dt.year
        )

        return db


class Mooney(CTUDataset):
    """
    The dataset describes a family composed of 86 people across 5 generations.\
    The family dataset includes 744 positive instances and 1488 randomly generated\
    negative instances.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Mooney_Family",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class MovieLens(CTUDataset):
    """
    MovieLens data set from the UC Irvine machine learning repository.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "imdb_MovieLens",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class MuskSmall(CTUDataset):
    """
    The Musk database describes molecules occurring in different conformations.\
    Each molecule is either musk or non-musk and one of the conformations determines\
    this property. Such a problem is known as a multiple-instance problem, and is modeled\
    by two tables molecule and conformation, joined by a one-to-many association.\
    Confirmation contains a molecule identifier plus 166 continuous features.\
    Molecule just contains the identifier and the class. There are two versions of\
    the dataset, MuskSmall, containing 92 molecules and 476 confirmations, and MuskLarge,\
    containing 102 molecules and 6598 confirmations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "MuskSmall",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class MuskLarge(CTUDataset):
    """
    The Musk database describes molecules occurring in different conformations.\
    Each molecule is either musk or non-musk and one of the conformations determines\
    this property. Such a problem is known as a multiple-instance problem, and is modeled\
    by two tables molecule and conformation, joined by a one-to-many association.\
    Confirmation contains a molecule identifier plus 166 continuous features.\
    Molecule just contains the identifier and the class. There are two versions of\
    the dataset, MuskSmall, containing 92 molecules and 476 confirmations, and MuskLarge,\
    containing 102 molecules and 6598 confirmations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "MuskLarge",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Mutagenesis(CTUDataset):
    """
    The dataset comprises of 230 molecules trialed for mutagenicity on\
    Salmonella typhimurium. A subset of 188 molecules is learnable using linear regression.\
    This subset was later termed the ”regression friendly” dataset. The remaining subset\
    of 42 molecules is named the ”regression unfriendly” dataset.\
    Note that authors use this dataset with a variable set of the background knowledge\
    and consequently, the reported accuracies do not have to be directly comparable.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "mutagenesis",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class NBA(CTUDataset):
    """
    A database with information about basketball matches from the National Basketball\
    Association. Lists Players, Teams, and matches with action counts for each player.
    
    TODO: contains value errors
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "NBA",
            cache_dir=cache_dir,
            time_col_dict={"Game": "Date"},
            keep_original_keys=False,
        )


class NCAA(CTUDataset):
    """
    NCAA Basketball Tournament.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "NCAA",
            cache_dir=cache_dir,
            time_col_dict={"seasons": "dayzero"},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        for table in db.table_dict.values():
            for fk, ref in table.fkey_col_to_pkey_table.items():
                if ref != "seasons":
                    continue
                table.df["date"] = pd.merge(
                    left=db.table_dict["seasons"].df,
                    right=table.df,
                    left_on="__PK__",
                    right_on=fk,
                    how="right",
                )["dayzero"]

                if "daynum" in table.df.columns:
                    table.df["date"] += pd.to_timedelta(table.df["daynum"], unit="D")
                    table.df.drop(columns=["daynum"], inplace=True)

                table.time_col = "date"
                break

        db.table_dict["target"].df.drop(columns=["pred", "team_id2_wins"], inplace=True)

        return db


class Northwind(CTUDataset):
    """
    The Northwind database contains the sales data for a fictitious company called\
    Northwind Traders, which imports and exports specialty foods from around the world.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "northwind",
            cache_dir=cache_dir,
            time_col_dict={"Orders": "OrderDate", "Employees": "HireDate"},
            keep_original_keys=False,
        )


class Pima(CTUDataset):
    """
    The National Institute of Diabetes and Digestive and Kidney Diseases conducted\
    a study on 768 adult female Pima Indians living near Phoenix.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Pima",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class PremiereLeague(CTUDataset):
    """
    A database with information about football matches from the UK Premier League.\
    Lists Players, Teams, and matches with action counts for each player.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "PremierLeague",
            cache_dir=cache_dir,
            time_col_dict={"Matches": "Date"},
            keep_original_keys=False,
        )


class Restbase(CTUDataset):
    """
    A database of restaurants in the San Francisco Bay Area.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "restbase",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["location"].df.drop(columns=["id_restaurant"], inplace=True)
        db.table_dict["generalinfo"].df.drop(columns=["id_restaurant"], inplace=True)

        return db


class Sakila(CTUDataset):
    """
    The Sakila sample database is designed to represent a DVD rental store.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "sakila",
            cache_dir=cache_dir,
            time_col_dict={"payment": "payment_date", "rental": "rental_date"},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        fk_name = "FK_film_film_id"
        db.table_dict["film_text"].df[fk_name] = db.table_dict["film_text"].df["__PK__"]
        db.table_dict["film_text"].fkey_col_to_pkey_table[fk_name] = "film"

        return db


class Sales(CTUDataset):
    """
    A simple artificial database in star schema.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "SalesDB",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class SameGen(CTUDataset):
    """
    Small database of family relations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Same_gen",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class SAP(CTUDataset):
    """
    Syntetic dataset containing information about sales of a Credit++.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "SAP",
            cache_dir=cache_dir,
            time_col_dict={"Sales": "EVENT_DATE"},
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        mailings_1_2 = db.table_dict["Mailings1_2"].df
        mailings_1_2.drop(columns=["KxIndex", "REFID"], inplace=True)
        fk_name = [
            k
            for k, v in db.table_dict["Mailings1_2"].fkey_col_to_pkey_table.items()
            if v == "Customers"
        ][0]

        mailings_3 = db.table_dict["mailings3"].df
        fk_col, _ = self._reindex_fk(
            {tn: t.df for tn, t in db.table_dict.items()},
            "mailings3",
            ["REFID"],
            "Customers",
            ["ID"],
        )
        mailings_3[fk_name] = fk_col
        mailings_3.drop(columns=["REFID"], inplace=True)
        mailings_3["__PK__"] += mailings_1_2["__PK__"].max() + 1

        db.table_dict["Customers"].df.drop(columns=["ID", "GEOID"], inplace=True)
        db.table_dict["Sales"].df.drop(columns=["EVENTID", "REFID"], inplace=True)
        db.table_dict["Demog"].df.drop(columns=["GEOID"], inplace=True)

        db.table_dict.pop("Mailings1_2")
        db.table_dict.pop("mailings3")

        mailings = pd.concat([mailings_1_2, mailings_3], axis=0)
        db.table_dict["Mailings"] = Table(
            df=mailings,
            fkey_col_to_pkey_table={fk_name: "Customers"},
            pkey_col="__PK__",
            time_col="REF_DATE",
        )

        return db


class Satellite(CTUDataset):
    """
    Communications satellite data.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "SAT",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        for table in db.table_dict.values():
            rm_fks = []
            for fk, ref in table.fkey_col_to_pkey_table.items():
                if ref in ["trfl", "class"]:
                    table.df.drop(columns=[fk], inplace=True)
                    rm_fks.append(fk)
            for fk in rm_fks:
                table.fkey_col_to_pkey_table.pop(fk)

        db.table_dict.pop("trfl")
        db.table_dict.pop("class")

        return db


class Seznam(CTUDataset):
    """
    Seznam.cz is a web portal and search engine in the Czech Republic.\
    The data represent online advertisement expenditures from Seznam's "wallet".
    
    Table description: 
        client: location and domain field of the client (anonymized)
        dobito: prepaid into a wallet in Czech currency 
        probehnuto: charged from the wallet in Czech currency
        probehnuto_mimo_penezenku: charged in Czech currency, but not from the wallet.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Seznam",
            cache_dir=cache_dir,
            time_col_dict={
                "dobito": "month_year_datum_transakce",
                "probehnuto": "month_year_datum_transakce",
                "probehnuto_mimo_penezenku": "Month_Year",
            },
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["probehnuto_mimo_penezenku"].df.rename(
            columns={"Month/Year": "Month_Year"}, inplace=True
        )

        return db


class SFScores(CTUDataset):
    """
    The San Francisco Dept. of Public Health's database of eateries, inspections\
    of those eateries, and violations found during the inspections. The scores of inspections\
    range from 1 to 100, where 100 means that the establishment meets all required standards.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "SFScores",
            cache_dir=cache_dir,
            time_col_dict={
                "inspections": "date",
                "violations": "date",
            },
            keep_original_keys=False,
        )


class Shakespeare(CTUDataset):
    """
    The Open Source Shakespeare is a collection of Shakespeare's complete works.\
    This is a much more interesting data set than some boring imaginary online retailer.\
    In this dataset, people die!
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Shakespeare",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class Stats(CTUDataset):
    """
    An anonymized dump of all user-contributed content on the Stats Stack Exchange network.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "stats",
            cache_dir=cache_dir,
            time_col_dict={
                "badges": "Date",
                "comments": "CreationDate",
                "postHistory": "CreationDate",
                "postLinks": "CreationDate",
                "posts": "CreaionDate",
                "users": "CreationDate",
                "votes": "CreationDate",
            },
            keep_original_keys=False,
        )


class StudentLoan(CTUDataset):
    """
    Student Loan contains data about students enrollment and employment status, and\
    the aim is to find rules that define a students' obligation for paying his/her loan back.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Student_loan",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class Thrombosis(CTUDataset):
    """
    PKDD'99 Medical dataset describes 41 patients with Thrombosis.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "medical",
            cache_dir=cache_dir,
            time_col_dict={"Laboratory": "Date"},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        db.table_dict["Examination"].df = db.table_dict["Examination"].df[
            db.table_dict["Examination"].df["Examination Date"].notna()
        ]

        return db


class Toxicology(CTUDataset):
    """
    Predictive Toxicology Challenge (2000) consists of more than three hundreds of organic\
    molecules marked according to their carcinogenicity on male and female mice and rats.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Toxicology",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class TPCC(CTUDataset):
    """
    TPC-C is the benchmark published by the Transaction Processing Performance Council\
    (TPC) for Online Transaction Processing (OLTP).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcc",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=True,
        )


class TPCD(CTUDataset):
    """
    TPC-D represents a broad range of decision support (DS) applications that require\
    complex, long running queries against large complex data structures.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcd",
            cache_dir=cache_dir,
            time_col_dict={
                "dss_lineitem": "l_shipdate",
                "dss_order": "o_orderdate",
            },
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        # Add missing foreign keys
        fk_col, fk_name = self._reindex_fk(
            {tn: t.df for tn, t in db.table_dict.items()},
            "dss_partsupp",
            ["ps_partkey"],
            "dss_part",
            ["p_partkey"],
        )
        db.table_dict["dss_partsupp"].fkey_col_to_pkey_table[fk_name] = "dss_part"
        db.table_dict["dss_partsupp"].df[fk_name] = fk_col

        # Drop redundant key columns
        db.table_dict["dss_lineitem"].df.drop(columns=["l_orderkey"], inplace=True)
        db.table_dict["dss_part"].df.drop(columns=["p_partkey"], inplace=True)
        db.table_dict["dss_partsupp"].df.drop(
            columns=["ps_partkey", "ps_suppkey"], inplace=True
        )
        db.table_dict["dss_region"].df.drop(columns=["r_regionkey"], inplace=True)
        db.table_dict["dss_nation"].df.drop(
            columns=["n_nationkey", "n_regionkey"], inplace=True
        )
        db.table_dict["dss_customer"].df.drop(
            columns=["c_custkey", "c_nationkey"], inplace=True
        )
        db.table_dict["dss_supplier"].df.drop(
            columns=["s_suppkey", "s_nationkey"], inplace=True
        )
        db.table_dict["dss_order"].df.drop(
            columns=["o_orderkey", "o_custkey"], inplace=True
        )

        return db


class TPCDS(CTUDataset):
    """
    TPC-DS is the new decision support benchmark that models several generally applicable\
    aspects of a decision support system, including queries and data maintenance. Although\
    the underlying business model of TPC-DS is a retail product supplier, the database\
    schema, data population, queries, data maintenance model and implementation rules have\
    been designed to be broadly representative of modern decision support systems.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpcds",
            cache_dir=cache_dir,
            time_col_dict={
                "store": "s_rec_start_date",
            },
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        # Replace date and time tables with simple datetime columns
        time_df = db.table_dict["time_dim"].df
        date_df = db.table_dict["date_dim"].df

        def to_time(x):
            return pd.Timedelta(
                hours=x["t_hour"], minutes=x["t_minute"], seconds=x["t_second"]
            )

        time_df["time_delta"] = db.table_dict["time_dim"].df.apply(to_time, axis=1)

        date_fk_dict = {
            tn: [
                fk for fk, fk_tn in t.fkey_col_to_pkey_table.items() if fk_tn == "date_dim"
            ]
            for tn, t in db.table_dict.items()
        }

        time_col_dict = {
            "store": "s_rec_start_date",
            "customer": "c_first_sales_date_sk",
            "web_page": "wp_creation_date_sk",
            "inventory": "inv_date_sk",
            "catalog_page": "cp_start_date_sk",
            "promotion": "p_start_date_sk",
            "web_site": "web_open_date_sk",
            "item": "i_rec_start_date",
            "store_returns": "sr_returned_date_sk",
            "catalog_sales": "cs_sold_date_sk",
            "call_center": "cc_open_date_sk",
            "web_sales": "ws_ship_date_sk",
        }

        time_fk_dict = {
            "store_returns": ("FK_time_dim_sr_return_time_sk", "sr_returned_date_sk"),
            "catalog_returns": ("FK_time_dim_cr_returned_time_sk", "cr_returned_date_sk"),
            "web_returns": ("FK_time_dim_wr_returned_time_sk", "wr_returned_date_sk"),
            "store_sales": ("FK_time_dim_ss_sold_time_sk", "ss_sold_date_sk"),
            "catalog_sales": ("FK_time_dim_cs_sold_time_sk", "cs_sold_date_sk"),
            "web_sales": ("FK_time_dim_ws_sold_time_sk", "ws_sold_date_sk"),
        }

        for t_name, fks in date_fk_dict.items():
            for fk in fks:
                db.table_dict[t_name].df[fk.removeprefix("FK_date_dim_")] = db.table_dict[
                    t_name
                ].df.join(date_df["d_date"], on=fk, how="left")["d_date"]
                db.table_dict[t_name].df.drop(columns=[fk], inplace=True)
                db.table_dict[t_name].fkey_col_to_pkey_table.pop(fk)
            time_col = time_col_dict.get(t_name, None)
            if time_col is not None:
                db.table_dict[t_name].time_col = time_col

        for t_name, (fk, date_col) in time_fk_dict.items():
            db.table_dict[t_name].df[date_col] += (
                db.table_dict[t_name]
                .df[["__PK__", fk]]
                .fillna(0)
                .join(time_df["time_delta"], on=fk, how="left")["time_delta"]
            )
            db.table_dict[t_name].df.drop(columns=[fk], inplace=True)
            db.table_dict[t_name].fkey_col_to_pkey_table.pop(fk)

        db.table_dict.pop("time_dim")
        db.table_dict.pop("date_dim")

        return db


class TPCH(CTUDataset):
    """
    TPC-H is the benchmark published by the Transaction Processing Performance Council\
    (TPC) for decision support.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "tpch",
            cache_dir=cache_dir,
            time_col_dict={"lineitem": "l_shipdate", "orders": "o_orderdate"},
            keep_original_keys=False,
        )


class Triazine(CTUDataset):
    """
    A pyrimidine QSAR dataset. The the goal is to predict the inhibition\
    of dihydrofolate reductase by pyrimidines.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Triazine",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class UWCSE(CTUDataset):
    """
    This dataset lists facts about the Department of Computer Science and Engineering\
    at the University of Washington (UW-CSE), such as entities (e.g., Student, Professor)\
    and their relationships (i.e. AdvisedBy, Publication).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "UW_std",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class VisualGenome(CTUDataset):
    """
    Visual Genome is a dataset, a knowledge base, an ongoing effort to connect\
    structured image concepts to language.
    
    TODO: add images
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "VisualGenome",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        for table in db.table_dict.values():
            rm_fks = []
            for fk, ref in table.fkey_col_to_pkey_table.items():
                if ref in ["ATT_CLASSES", "OBJ_CLASSES", "PRED_CLASSES"]:
                    col = ref.removesuffix("ES")

                    table.df[col] = pd.merge(
                        left=db.table_dict[ref].df,
                        right=table.df,
                        left_on="__PK__",
                        right_on=fk,
                        how="right",
                    )[col]
                    table.df.drop(columns=[fk], inplace=True)
                    rm_fks.append(fk)

            for fk in rm_fks:
                table.fkey_col_to_pkey_table.pop(fk)

        db.table_dict.pop("ATT_CLASSES")
        db.table_dict.pop("OBJ_CLASSES")
        db.table_dict.pop("PRED_CLASSES")

        return db


class VOC(CTUDataset):
    """
    VOC database provides a peephole view into the administrative system of an early\
    multi-national company, the Vereenigde geoctrooieerde Oostindische Compagnie (VOC for\
    short - The (Dutch) East Indian Company) established on March 20, 1602.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "voc",
            cache_dir=cache_dir,
            time_col_dict={"voyages": "departure_date"},
            keep_original_keys=False,
        )

    def customize_db(self, db: Database) -> Database:
        dtcols = db.table_dict["voyages"].df.select_dtypes(include="datetime")
        dtcols[dtcols < pd.Timestamp("1300-01-01")] = pd.NaT
        dtcols[dtcols > pd.Timestamp("1900-01-01")] = pd.NaT
        min_date = dtcols.min().min()
        max_date = dtcols.max().max()
        delta: pd.Timedelta
        if min_date < TIMESTAMP_MIN:
            delta = TIMESTAMP_MIN - min_date
            delta = pd.Timedelta(days=delta.days + 1)
        elif max_date > TIMESTAMP_MAX:
            delta = TIMESTAMP_MAX - max_date
            delta = pd.Timedelta(days=delta.days - 1)
        dtcols += delta.to_numpy().astype(np.dtype("timedelta64[D]"))

        db.table_dict["voyages"].df[dtcols.columns] = dtcols

        return db


class Walmart(CTUDataset):
    """
    The sales of 111 potentially weather-sensitive products (like umbrellas, bread, and milk)\
    around the time of major weather events at 45 of their retail locations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "Walmart",
            cache_dir=cache_dir,
            time_col_dict={"weather": "date"},
            keep_original_keys=True,
        )

    def customize_db(self, db: Database) -> Database:
        # Convert sunrise and sunset to datetime
        weather_df = db.table_dict["weather"].df
        weather_df["sunrise"] = weather_df["date"] + weather_df["sunrise"].fillna(
            pd.Timedelta(hours=6)
        )
        weather_df["sunset"] = weather_df["date"] + weather_df["sunset"].fillna(
            pd.Timedelta(hours=18)
        )
        db.table_dict["weather"].df = weather_df

        table = db.table_dict.pop("train")
        table.time_col = "date"
        db.table_dict["train_table"] = table

        return db


class WebKP(CTUDataset):
    """
    The WebKB dataset consists of 877 scientific publications classified into one of five\
    classes. The citation network consists of 1608 links. Each publication in the dataset\
    is described by a 0/1-valued word vector indicating the absence/presence of the\
    corresponding word from the dictionary. The dictionary consists of 1703 unique words.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "WebKP",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )


class World(CTUDataset):
    """
    A database of 239 states and their cities.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            "world",
            cache_dir=cache_dir,
            time_col_dict={},
            keep_original_keys=False,
        )
