import logging
import os
from pathlib import Path

import pandas as pd
import pytest

log_fmt = "[%(asctime)s %(levelname)-8s] [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"  # noqa
logging.basicConfig(level=logging.DEBUG, format=log_fmt)

logger = logging.getLogger(__name__)

import os
from pathlib import Path
try:
    # '.' if the path is to current folder
    os.chdir(Path(os.path.join(os.getcwd(), '.')).parents[0])
    print(os.getcwd())
except:
    pass

@pytest.fixture(scope="session")
def data():
    try:
        pardir = Path(__file__).parents[0]
        file_path = os.path.join(pardir, "test_data.csv")
        answers = pd.read_csv(file_path, index_col=None)
    except OSError as err:
        logger.error(
            "{}: cannot locate excel file check folder for feature_df.csv".format(  # noqa
                type(err)
            ),
            exc_info=True,
        )
        raise
    
    data_sub=answers[(answers.Species=='Iris-setosa') | (answers.Species=='Iris-virginica')]
    data_sub.loc[(data_sub['Species']=='Iris-setosa'),'classify']=1
    data_sub=data_sub.fillna(0)
    return data_sub


