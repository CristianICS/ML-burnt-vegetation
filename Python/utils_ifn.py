"""Handle Spain FNI databases"""
from pathlib import Path
import re
from glob import glob
from dbfread import DBF # type: ignore
import pyodbc # type: ignore
import pandas as pd # type: ignore

class Dbf:

    def __init__(self, path: str):
        """Opens DBF database as pandas dataframe"""
        # Store path
        self.path = path
        # Open database
        dbf = DBF(path)
        # Transform above db into a pandas dataframe
        df = pd.DataFrame(iter(dbf))
        # Transform string numbers columns to numeric and store df
        self.df = df.apply(self.convert_numeric)

    def convert_numeric(self, s: pd.Series):
        """
        Convert DF column to numeric and remain unchanged if it is not
        possible.

        This function is inserted inside pandas.DataFrame.apply method.
        """
        # Replace values '' by 0
        s.replace('', '0', inplace=True)
        return pd.to_numeric(s, downcast='integer', errors='coerce').fillna(s)

class Second:

    # Store filtered DATEST tables
    datest_df = None

    # Store shrub info
    matorr_df = None

    # Store stats by tree
    ifl_df = None

    def __init__(self, fpath: Path):
        """
        Init the class containing the second NFI database tables.
        
        Add the main folder directory where all the tables are stored.

        :self: Reference to the current instance of the class.
        :fpath: Location of the NFI tables' folder directory.
        """
        self.main_folder = fpath
        self.open_datest()
        self.open_matorr()
        self.open_ifl()
        self.open_piesma()

    def open_tables(self, table_paths: list) -> pd.DataFrame:
        """
        Iterate over DBF table paths, open it and save it inside a variable
        """
        # Store each table inside the following variable
        tables = False

        if len(table_paths) == 0:
            raise ValueError("NFIv2: There is no tables available.")

        # Open each table and then remove from table_paths list
        while len(table_paths) > 0:

            # Init table
            pandas_table = Dbf(Path(self.main_folder, table_paths[0])).df

            # Add each table information inside one main table
            if (type(tables) != type(False)):
                tables = pd.concat([tables, pandas_table])
            else:
                tables = pandas_table

            # Remove item from the datest list
            table_paths.pop(0)

        return tables

    def open_datest(self):
        """Open all DATEST tables."""
        # Retrieve all tables
        tbls = list(self.main_folder.glob('**/DATEST*.DBF'))

        # Store above tables inside a unique pandas df
        pandas_df = self.open_tables(tbls)
        # Store the above dataframe inside class variable
        self.datest_df = pandas_df

    def open_matorr(self):
        """Open MATORR tables with shrub data"""
        # Retrieve all tables
        tbls = list(self.main_folder.glob('**/MATORR*.DBF'))
        # Store above tables inside a unique pandas df
        pandas_df = self.open_tables(tbls)
        # Store the above dataframe inside a variable
        self.matorr_df = pandas_df

    def matorr_stats(self):
        """
        Create new columns with new custom stats.

        Each inventory plot contains as number of rows as shrubs in it.
        
        So, when metrics are grouped by ESTADILLO and PROVINCIA, the produced
        dataframe has all the shrub metrics by plot.
        """
        # Sum FCC values by plot
        fccsum = pd.NamedAgg(column="FRACCAB", aggfunc="sum")
        # Compute mean FCC values by plot
        fccmean = pd.NamedAgg(column="FRACCAB", aggfunc="mean")
        # Compute mean elevation by plot
        hmean = pd.NamedAgg(column="ALTUMED", aggfunc="mean")

        # When groupby is applied, the ESTADILLO column is set as index
        df_group = self.matorr_df.groupby(['ESTADILLO', 'PROVINCIA']).agg(
            mat_fcc_sum = fccsum,
            mat_fcc_mean = fccmean,
            mat_hm = hmean
        )
        target_cols = [
           'ESTADILLO', 'PROVINCIA', 'mat_fcc_sum', 'mat_fcc_mean', 'mat_hm']
        # Return data with ESTADILLO as column (applying reset_index)
        self.matorr_df = df_group.reset_index()[target_cols]
    
    def open_ifl(self):
        """Open tables with tree data"""
        # Retrieve all ifl00BD tables
        tbls = list(self.main_folder.glob('**/IIFL03BD*.DBF'))
        # Store above tables inside a unique pandas df
        pandas_df = self.open_tables(tbls)
        # Rename some columns to perform later join on DATEST
        rename = {
            "CESTADILLO": "ESTADILLO",
            "CPROVINCIA": "PROVINCIA"
        }
        pandas_df = pandas_df.rename(columns=rename)
        # Store the above dataframe inside a variable
        self.ifl_df = pandas_df

    def ifl_stats(self):
        """Compute TREE stats by SPECIE."""
        # Value with the number of trees
        m_trees = pd.NamedAgg(column="NARBOLES", aggfunc="mean")
        # Mean of the AREAB of every species (divided by CCDIAMETR)
        mean_ab = pd.NamedAgg(column="NAREAB", aggfunc="mean")

        # Columns to aggregate the info
        gcols = ['ESTADILLO', 'PROVINCIA', 'CESPECIE']
        df_group = self.ifl_df.groupby(gcols).agg(
            n_arb = m_trees,
            mean_areab = mean_ab
        )

        target_cols = gcols + ['n_arb', 'mean_areab']
        self.ifl_df = df_group.reset_index()[target_cols]

    def open_piesma(self):
        """Open all PIESMA tables."""
        # Retrieve all tables
        tbls = list(self.main_folder.glob('**/PIESMA*.DBF'))

        # Store above tables inside a unique pandas df
        pandas_df = self.open_tables(tbls)
        # Store the above dataframe inside class variable
        self.piesma_df = pandas_df

    def piesma_stats(self):
        """Compute mean Diameter"""
        # Perform the mean of the two DIAMETRO columns
        mean_dn = (self.piesma_df['DIAMETRO1']
            .add(self.piesma_df['DIAMETRO2']).divide(2))
        # Store it inside the dataframe
        self.piesma_df['tree_mean_dn'] = mean_dn

        # Get the mean by plot and species
        esp_dn_mean = pd.NamedAgg('tree_mean_dn', 'mean')
        # Columns to aggregate the info
        gcols = ['ESTADILLO', 'PROVINCIA', 'ESPECIA12']
        df_group = self.piesma_df.groupby(gcols).agg(mean_dn = esp_dn_mean)
        # Export the df with the stats and target columns
        target_cols = gcols + ['mean_dn']

        self.piesma_df = df_group.reset_index()[target_cols]

class Access:

    def __init__(self, dbpath: Path | str):
        """Handle MDB and ACCDB databases."""
        # Store path
        self.path = str(dbpath)
        # Open the database
        self.connect()

    def connect(self):
        """
        Create a connection into the database

        Note: Cursors created from the same connection are not isolated, i.e.,
        any changes done to the database by a cursor are immediately visible
        by the other cursors.
        https://github.com/mkleehammer/pyodbc/wiki/Cursor
        """
        conn = pyodbc.connect(
            DRIVER = '{Microsoft Access Driver (*.mdb, *.accdb)}',
            DBQ = self.path
        ) # type: pyodbc.Cursor
        # Store the object with the connection
        self.conn = conn

    def filter(self,
             tname: str,
             where: str = '',
             cols: list = None,
             join: str = None,
             groupby: str = None) -> pd.DataFrame:
        """
        Apply SQL query to MDB/ACCDB database.
    
        Note: Avoid the use of double quotes (") for strings inside 'WHERE'
        statement. The strings must be surrounded only by single quotes: ('a').

        This function could be used to retrieve all the data inside a table
        like this: self.filter('tname', '')

        A join could be inserted to retrieve the data from two tables:
        'INNER JOIN table1 ON table1.col1=table2.col1'

        :tname: Table name.
        :where: SQL filter.
        :columns: Columns to fetch.
        :join: JOIN clause
        :groubpy: GROUP BY clause, e.g., 'col1,col2'
        """
        # Handle SELECT query statement
        if type(cols) == type(None):
            select_sql = "SELECT *"
        else:
            select_sql = "SELECT {}".format(','.join(cols))

        # Handle FROM statement
        from_sql = f"FROM {tname}"

        # Handle WHERE statement
        if len(where) == 0:
            where_sql = ''
        else:
            where_sql = f"WHERE {where}"

        # Handle JOIN statement
        if type(join) == type(None):
            query = ' '.join([select_sql, from_sql, where_sql])
        else:
            query = ' '.join([select_sql, from_sql, join, where_sql])

        # Handle GROUP BY statement
        if type(groupby) != type(None):
            query = f'{query} GROUP BY {groupby}'

        # Create a new cursor
        with self.conn.cursor() as crs:
            # Execute the prior query
            crs.execute(query)
            # Use custom class to handle the response, converted it into pandas
            qurey_result = querySQL(crs)
        
        return qurey_result.to_pandas()

class querySQL:

    def __init__(self, query_result):
        """
        Handle query result of a pyodbc fetchall method.

        Save the columns and values from the query. The fetchall method returns
        a list like the following, with no column names.

        [('col1_val1', 'col2_val2'), ('col1_val3', 'col2_val4')]

        :query_result: pyodbc.cursor object with active query in it.
        """
        # Create a list with the resulted query columns
        cols = [c[0] for c in query_result.description]
        # Create a list of tuples.
        # Each tuple contains a number of values as number of fetched columns 
        vals = query_result.fetchall()
        # Save the two objects
        self.cols = cols
        self.vals = vals

    def _to_dict(self):
        """Convert query to a dict merging values and columns together."""
        # Save the key:value pairs inside a list.
        # The goal is transforming it into a pandas dataframe.
        d = [dict(map(lambda i,j: (i,j), self.cols, row)) for row in self.vals]
        return d
  
    def to_pandas(self):
        """Convert dict (from previos sql query) into a pandas dataframe"""
        query_dict = self._to_dict()
        return pd.DataFrame(query_dict)

class Modern:

    def __init__(self, fpath: Path):
        """
        Init the class containing NFI3 and NFI4 databases.
        
        Add the main folder directory with all the NFI3 tables are stored.

        :self: Reference to the current instance of the class.
        :fpath: Location of the NFI2 tables' folder directory.
        """
        # Save the main folder, it will use to open the databases later
        self.main_folder = fpath
        # Retrieve all Ifn tables (no matter the extension)
        paths_ifn = [str(p) for p in fpath.glob('**/Ifn*')]
        # Check the valid extension ones and save them
        extensions = r'.*\.(mdb|accdb)'
        self.ifn_paths = self.glob_re(extensions, paths_ifn)
        # Retrieve all Sig tables
        paths_sig = [str(p) for p in fpath.glob('**/Sig*')]
        self.sig_paths = self.glob_re(extensions, paths_sig)

    def glob_re(self, pattern, strings) -> list:
        """Use regular expression to scan multiple paths."""
        filtered_p = [f for f in filter(re.compile(pattern).match, strings)]
        if len(filtered_p) == 0:
            raise ValueError("No valid paths have been located.")
        else:
            return filtered_p

    def get_province_code(self, dbpath: Path | str):
        """
        Extract province code from database name
        Do not use \\d character to avoid selecting 00 in ifl00DB22.
        """
        if type(dbpath) == str:
            dbpath = Path(dbpath)
        prov = re.search(r'.*([1-9]\d).*', dbpath.stem, re.IGNORECASE).group(1)
        return int(prov)

    def open(self, tname, table = "ifn", **kwargs):
        """
        Iterative filter, append and show the province databases
        
        :tname: Table name to retrieve from the databases.
        :table: ifn or sig.
        :**kwargs: Options to pass to Access filter method.
        """
        if table == "sig":
            tables = self.sig_paths
        else:
            tables = self.ifn_paths

        if len(tables) == 0:
            raise ValueError(f"There is no {table}* tables available.")
        
        # Store each table inside the following variable
        tables_df = False

        # Open each database and apply the filter
        # If no parameters, the filter will return all the table
        for t_path in tables:

            # Open connection
            db = Access(t_path)
            # Filter the data and return a pandas database
            df = db.filter(tname, **kwargs)

            # Handle the province code
            if 'Provincia' not in df.columns.to_list():
                # Show a warning message
                msg = (
                    f"ResourceWarning: The table {t_path.name}",
                    "has not 'Provincia' column.",
                    "It has been created automatically.")
                print(' '.join(msg))

            prov_code = self.get_province_code(t_path)
            df['Provincia'] = prov_code

            # Add each table information inside one main table
            if (type(tables_df) != type(False)):
                tables_df = pd.concat([tables_df, df])
            else:
                tables_df = df

        return tables_df
