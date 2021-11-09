# TFM

##### Código para el TFM de "Impactos de los incendios forestales en España y en grandes Ciudades"  de los alumnos del master de Business Inteligent & data science del curso 2020-2021

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "164e6e5e",
   "metadata": {},
   "source": [
    "Importar librería y lectura de datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "6c4b1933",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from matplotlib.pyplot import plot, title\n",
    "from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve\n",
    "from sklearn.utils import resample\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "96829585",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"fires-all.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "11b3e7ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 82640 entries, 0 to 82639\n",
      "Data columns (total 21 columns):\n",
      " #   Column           Non-Null Count  Dtype  \n",
      "---  ------           --------------  -----  \n",
      " 0   id               82640 non-null  int64  \n",
      " 1   superficie       82640 non-null  float64\n",
      " 2   fecha            82640 non-null  object \n",
      " 3   lat              82616 non-null  float64\n",
      " 4   lng              82616 non-null  float64\n",
      " 5   latlng_explicit  82640 non-null  int64  \n",
      " 6   idcomunidad      82640 non-null  int64  \n",
      " 7   idprovincia      82640 non-null  int64  \n",
      " 8   idmunicipio      82640 non-null  int64  \n",
      " 9   municipio        82640 non-null  object \n",
      " 10  causa            82640 non-null  int64  \n",
      " 11  causa_supuesta   46465 non-null  float64\n",
      " 12  causa_desc       82640 non-null  int64  \n",
      " 13  muertos          2724 non-null   float64\n",
      " 14  heridos          3071 non-null   float64\n",
      " 15  time_ctrl        82640 non-null  int64  \n",
      " 16  time_ext         82640 non-null  int64  \n",
      " 17  personal         82640 non-null  int64  \n",
      " 18  medios           82640 non-null  int64  \n",
      " 19  gastos           11624 non-null  float64\n",
      " 20  perdidas         34349 non-null  float64\n",
      "dtypes: float64(8), int64(11), object(2)\n",
      "memory usage: 13.2+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cfd7b40d",
   "metadata": {},
   "source": [
    "Se eliminan las variables no necesarias para la predicción."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "ec031393",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop([\"id\", \"fecha\", \"lat\", \"lng\", \"latlng_explicit\", \"municipio\",\n",
    "        \"causa_supuesta\", \"causa_desc\"], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe65bdd1",
   "metadata": {},
   "source": [
    "Se reasignan los tipos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "7b9636f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"idcomunidad\"] = df[\"idcomunidad\"].astype(str)\n",
    "df[\"idprovincia\"] = df[\"idprovincia\"].astype(str)\n",
    "df[\"idmunicipio\"] = df[\"idmunicipio\"].astype(str)\n",
    "df[\"causa\"] = df[\"causa\"].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "8df8d432",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 82640 entries, 0 to 82639\n",
      "Data columns (total 13 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   superficie   82640 non-null  float64\n",
      " 1   idcomunidad  82640 non-null  object \n",
      " 2   idprovincia  82640 non-null  object \n",
      " 3   idmunicipio  82640 non-null  object \n",
      " 4   causa        82640 non-null  object \n",
      " 5   muertos      2724 non-null   float64\n",
      " 6   heridos      3071 non-null   float64\n",
      " 7   time_ctrl    82640 non-null  int64  \n",
      " 8   time_ext     82640 non-null  int64  \n",
      " 9   personal     82640 non-null  int64  \n",
      " 10  medios       82640 non-null  int64  \n",
      " 11  gastos       11624 non-null  float64\n",
      " 12  perdidas     34349 non-null  float64\n",
      "dtypes: float64(5), int64(4), object(4)\n",
      "memory usage: 8.2+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13cf1458",
   "metadata": {},
   "source": [
    "Tratamiento de los nulos de las variables \"muertos\" y \"heridos\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "5c8297ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.muertos = df.muertos.fillna(0)\n",
    "df.heridos = df.heridos.fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "9a53de39",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.muertos = df.muertos.astype(int)\n",
    "df.heridos = df.heridos.astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67ff44a7",
   "metadata": {},
   "source": [
    "Se elimina la variable \"gastos\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2baea3b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"gastos\", axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "b6ad089b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 82640 entries, 0 to 82639\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   superficie   82640 non-null  float64\n",
      " 1   idcomunidad  82640 non-null  object \n",
      " 2   idprovincia  82640 non-null  object \n",
      " 3   idmunicipio  82640 non-null  object \n",
      " 4   causa        82640 non-null  object \n",
      " 5   muertos      82640 non-null  int64  \n",
      " 6   heridos      82640 non-null  int64  \n",
      " 7   time_ctrl    82640 non-null  int64  \n",
      " 8   time_ext     82640 non-null  int64  \n",
      " 9   personal     82640 non-null  int64  \n",
      " 10  medios       82640 non-null  int64  \n",
      " 11  perdidas     34349 non-null  float64\n",
      "dtypes: float64(2), int64(6), object(4)\n",
      "memory usage: 7.6+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fca94e63",
   "metadata": {},
   "source": [
    "Tratamiento de los nulos de la variable \"perdidas\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "7bb5e84f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def tidy_corr_matrix(corr_mat):\n",
    "    '''\n",
    "    Función para convertir una matriz de correlación de pandas en formato tidy\n",
    "    '''\n",
    "    corr_mat = corr_mat.stack().reset_index()\n",
    "    corr_mat.columns = ['variable_1','variable_2','r']\n",
    "    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]\n",
    "    corr_mat['abs_r'] = np.abs(corr_mat['r'])\n",
    "    corr_mat = corr_mat.sort_values('abs_r', ascending=False)\n",
    "    \n",
    "    return(corr_mat)\n",
    "corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "6445b593",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))\n",
    "\n",
    "sns.heatmap(\n",
    "    corr_matrix,\n",
    "    annot     = True,\n",
    "    cbar      = False,\n",
    "    annot_kws = {\"size\": 8},\n",
    "    vmin      = -1,\n",
    "    vmax      = 1,\n",
    "    center    = 0,\n",
    "    cmap      = sns.diverging_palette(20, 220, n=200),\n",
    "    square    = True,\n",
    "    ax        = ax\n",
    ")\n",
    "\n",
    "ax.set_xticklabels(\n",
    "    ax.get_xticklabels(),\n",
    "    rotation = 45,\n",
    "    horizontalalignment = 'right',\n",
    ")\n",
    "\n",
    "ax.tick_params(labelsize = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "88af5e34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZIAAAEJCAYAAAC+I6F6AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAArbUlEQVR4nO3de5gdVZ3u8e9LaKCBQLgEhARMBAwDXhJoAUU9oI7JeCMKDGFQcGSMclBhUBQ8KpcZFA+jeBBhBkURRSByV0RAIOIFgh0SCAGiUQIkZCBMLgQTyYXf+aPWJrs7+1Kd3bt7V+/38zz76dqrbr/q6q5VtdaqtRQRmJmZbarNBjsAMzMrNmckZmbWEGckZmbWEGckZmbWEGckZmbWEGckZmbWkKZlJJK2kvSApIckzZV0TkrfUdKdkv6Ufu5Qts6ZkuZLmidpYln6gZLmpHkXSVJK31LStSl9hqQxzToeMzOrrJlPJC8B74iINwLjgUmSDgHOAO6KiH2Au9J3JO0HTAH2ByYBl0galrZ1KTAV2Cd9JqX0E4FlEbE3cCHw9SYej5mZVbB5szYc2ZuOL6avHekTwBHAYSn9h8B04Asp/ZqIeAl4QtJ84CBJC4DtIuI+AElXApOB29I6Z6dtXQdcLElR4y3LnXfeOcaMGdMfh2hm1jZmzpz5fESMrDSvaRkJQHqimAnsDXwnImZI2jUiFgNExGJJu6TFRwH3l62+MKWtTdO900vrPJ22tU7SCmAn4PlqMY0ZM4bu7u6Gj83MrJ1IerLavKZWtkfE+ogYD4wme7p4XY3FVWkTNdJrrdNzw9JUSd2SupcsWVInajMz64sBabUVEcvJirAmAc9K2g0g/XwuLbYQ2KNstdHAMyl9dIX0HutI2hzYHlhaYf+XRURXRHSNHFnxyczMzDZRM1ttjZQ0Ik13Au8CHgduAU5Ii50A3JymbwGmpJZYY8kq1R9IxWArJR2SWmsd32ud0raOAu6uVT9iZmb9r5l1JLsBP0z1JJsB0yLi55LuA6ZJOhF4CjgaICLmSpoGPAqsA06OiPVpWycBVwCdZJXst6X0y4EfpYr5pWStvszMbACp3W7gu7q6wpXtZmZ9I2lmRHRVmtfUVltmRXbTrEVccPs8nlm+mt1HdHL6xHFMnjCq/opmbcYZiVkFN81axJk3zGH12qx0ddHy1Zx5wxwAZyZmvbivLbMKLrh93iuZSMnqteu54PZ5gxSRWetyRmJWwTPLV/cp3aydOSMxq2D3EZ19SjdrZ85IzCo4feI4OjuG9Ujr7BjG6RPHDVJEZq3Lle1mFZQq1N1qy6w+ZyRmVUyeMMoZh1kOLtoyM7OGOCMxM7OGOCMxM7OGOCMxM7OGOCMxM7OGOCMxM7OGOCMxM7OGOCMxM7OG5HohUdIWwGvT13kRsbZ5IZmZWZHUzUgkHQb8EFgACNhD0gkRcW9TIzMzs0LI80TyDeDdETEPQNJrgauBA5sZmJmZFUOeOpKOUiYCEBF/BDqaF5KZmRVJnieSbkmXAz9K348DZjYvJDMzK5I8GclJwMnAZ8jqSO4FLmlmUGZmVhx1M5KIeAn4ZvqYmZn1UDUjkTQtIv5R0hwges+PiDc0NTIzMyuEWk8kp6Sf7xuIQMzMrJiqttqKiMXp55OVPvU2LGkPSfdIekzSXEmnpPSzJS2SNDt93lO2zpmS5kuaJ2liWfqBkuakeRdJUkrfUtK1KX2GpDEN/C7MzGwT1CraWkmFIq2SiNiuzrbXAZ+NiAclDQdmSrozzbswIv6j1/72A6YA+wO7A7+S9NqIWA9cCkwF7gd+AUwCbgNOBJZFxN6SpgBfB46pE5eZmfWjqhlJRAwHkHQu8N9kzX9F1vx3eL0Npyea0lPNSkmPAbUGwD4CuCZV7j8haT5wkKQFwHYRcV+K50pgMllGcgRwdlr/OuBiSYqIqhmgmZn1rzwvJE6MiEsiYmVEvBARlwJH9mUnqchpAjAjJX1K0sOSvi9ph5Q2Cni6bLWFKW1Umu6d3mOdiFgHrAB26ktsZmbWmDwZyXpJx0kaJmkzSccB6/PuQNK2wPXAqRHxAlkx1V7AeLInlm+UFq2wetRIr7VO7ximSuqW1L1kyZK8oZuZWQ55MpJ/Av4ReDZ9jk5pdUnqIMtEroqIGwAi4tmIWB8RLwPfBQ5Kiy8E9ihbfTTwTEofXSG9xzqSNge2B5b2jiMiLouIrojoGjlyZJ7Qzcwsp7oZSUQsiIgjImLniBgZEZMjYkG99VLLqsuBxyLim2Xpu5Ut9kHgkTR9CzAltcQaC+wDPJDqWlZKOiRt83jg5rJ1TkjTRwF3u37EzGxg5elGfiTwcWBM+fIR8bE6qx4KfASYI2l2SvsicKyk8WRFUAuAT6TtzZU0DXiUrMXXyanFFmTdtFwBdJJVst+W0i8HfpQq5peStfoyM7MBpHo38JJ+D/yGrKPGV+pGIuL65obWHF1dXdHd3T3YYZiZFYqkmRHRVWlenk4bt46IL/RzTGZmNkTkqWz/efnb52ZmZuXyZCSnkGUmf5P0gqSVkl5odmBmZlYMebqRr/sWu5mZta+6TyTKfFjSl9P3PSQdVG89MzNrD3mKti4B3syGlxBfBL7TtIjMzKxQ8rTaOjgiDpA0CyAilknaoslxmZlZQeR5IlkraRipD6v0guLLTY3KzMwKI09GchFwI7CLpPOA3wJfbWpUZmZWGHlabV0laSbwTrLedidHxGNNj8zMzAohT19bhwBzI+I76ftwSQdHxIw6q5qZWRvIU7R1KVlLrZK/pjQzM7NcGUmPoWvTOCJ5WnuZmVkbyJOR/EXSZyR1pM8pwF+aHZiZmRVDnozkk8BbgEVkIxIeDExtZlBmZlYceVptPYcHjDIzsyqaOUKimZm1gTyV5jeTjZD4K8pGSDQzMwOPkGhmZg3yCIlmZtaQvoyQuNojJJqZWW8eIdHMzBpSNSORtG9EPC7pgErzI+LB5oVlZmZFUeuJ5DSyFw+/UWFeAO9oSkRmZlYoVTOSiJiafh4+cOGYmVnR5KlsNzMzq6ppGYmkPSTdI+kxSXNTZ49I2lHSnZL+lH7uULbOmZLmS5onaWJZ+oGS5qR5F0lSSt9S0rUpfYakMc06HjMzq6yZTyTrgM9GxN8BhwAnS9oPOAO4KyL2Ae5K30nzpgD7A5OAS9JY8ZCNfzIV2Cd9JqX0E4FlEbE3cCHw9SYej5mZVVCr1VbF1lol9VptRcRiYHGaXinpMWAUcARwWFrsh8B04Asp/ZqIeAl4QtJ84CBJC4DtIuK+FNeVwGTgtrTO2Wlb1wEXS+oxfoqZmTVXrVZblVprlfSp1VYqcpoAzAB2TZkMEbFY0i5psVHA/WWrLUxpa9N07/TSOk+nba2TtALYCXg+b2xmZtaYWq22+qW1lqRtgeuBUyPihVS9UXHRSmHUSK+1Tu8YppLGUNlzzz3rhWxmZn1Qt45E0taSviTpsvR9H0nvy7NxSR1kmchVEXFDSn5W0m5p/m7Acyl9IbBH2eqjgWdS+ugK6T3WkbQ5sD2wtHccEXFZRHRFRNfIkSPzhG5mZjnlqWz/AbCGbJREyC7e/15vpdSy6nLgsYj4ZtmsW4AT0vQJZN3Ul9KnpJZYY8kq1R9IxWArJR2Stnl8r3VK2zoKuNv1I2ZmAytPN/J7RcQxko4FiIjVqlE+VeZQ4CPAHEmzU9oXgfOBaZJOBJ4Cjk7bnStpGvAoWYuvkyOiNP7JScAVQCdZJfttKf1y4EepYn4pHsnRzGzA5clI1kjqJNU9SNoLeKneShHxWyrXYQC8s8o65wHnVUjvBl5XIf1vpIzIzMwGR56M5Czgl8Aekq4ie9L4aDODMjOz4sjTjfydkh4ke6lQwCkR4ea1ZmYG5HsiISL+B7i1ybGYmVkBudNGMzNriDMSMzNrSK6irdR54q7ly0fEU80KyszMiqNuRiLp02Qtt54FXk7JAbyhiXGZmVlB5HkiOQUYlyrczczMeshTR/I0sKLZgZiZWTHleSL5CzBd0q2UvdHeq/8sMzNrU3kykqfSZ4v0MTMze0WeN9vPAZC0TUT8tfkhmZlZkeQZj+TNkh4FHkvf3yjpkqZHZmZmhZCnsv1bwETgfwAi4iHg7U2MyczMCiTXm+0R8XSvpPUVFzQzs7aTp7L9aUlvAULSFsBnSMVcZmZmeZ5IPgmcDIwiG2Z3fPpuZmaWq9XW88BxAxCLmZkVkHv/NTOzhjgjMTOzhjgjMTOzhuQdj+S9wP7AVqW0iDi3WUGZmVlx5Hmz/T+BY4BPAwKOBl7d5LjMzKwg8hRtvSUijgeWpX633gzs0dywzMysKPJkJKvTz1WSdgfWAmObF5KZmRVJnozk55JGABcADwILgGvqrSTp+5Kek/RIWdrZkhZJmp0+7ymbd6ak+ZLmSZpYln6gpDlp3kWSlNK3lHRtSp8haUzegzYzs/5TNyOJiH+LiOURcT1Z3ci+EfHlHNu+AphUIf3CiBifPr8AkLQfMIWsQn8ScImkYWn5S4GpwD7pU9rmiWTFbXsDFwJfzxGTmZn1s6qttiR9qMY8IuKGWhuOiHv78JRwBHBNRLwEPCFpPnCQpAXAdhFxX9rvlcBk4La0ztlp/euAiyUpIiLnPs3MrB/Uav77/vRzF+AtwN3p++HAdKBmRlLDpyQdD3QDn42IZWT9eN1ftszClLY2TfdOJ/18GiAi1klaAewEPL+JcZmZ2SaoWrQVEf8cEf8MBLBfRBwZEUeSFT9tqkuBvcg6flwMfCOlq1IINdJrrbMRSVMldUvqXrJkSZ8CNjOz2vJUto+JiMVl358FXrspO4uIZyNifUS8DHwXOCjNWkjPJsWjgWdS+ugK6T3WkbQ5sD2wtMp+L4uIrojoGjly5KaEbmZmVeTJSKZLul3SRyWdANwK3LMpO5O0W9nXDwKlFl23AFNSS6yxZJXqD6QMbKWkQ1JrreOBm8vWOSFNHwXc7foRM7OBl6cb+U+live3paTLIuLGeutJuho4DNhZ0kLgLOAwSePJiqAWAJ9I+5graRrwKLAOODkiSqMwnkTWAqyTrJL9tpR+OfCjVDG/lKzVl5mZDTC12018V1dXdHd3D3YYZmaFImlmRHRVmpenr61DJP1B0ouS1khaL+mF/g/TzMyKKE8dycXAscCfyIqX/gX4djODMjOz4sjVjXxEzJc0LNVb/EDS75scl5mZFUSejGSVpC2A2ZL+L9n7H9s0NywzMyuKPEVbHwGGAZ8C/kr27saRzQzKzMyKI0/z3yfT5GrgnOaGY2ZmRZOn1db7JM2StFTSC5JWutWWmZmV5Kkj+RbwIWCO3xw3s1Zx06xFXHD7PJ5ZvprdR3Ry+sRxTJ4wqv6K1u/yZCRPA484EzGzVnHTrEWcecMcVq/NOsBYtHw1Z94wB8CZySDIk5F8HviFpF8DL5USI+KbTYvKzKyGC26f90omUrJ67XouuH2eM5JBkCcjOQ94EdgK2KK54ZiZ1ffM8tV9SrfmypOR7BgR7256JGZmOe0+opNFFTKN3Ud0DkI0luc9kl9JckZiZi3j9Inj6OwY1iOts2MYp08cN0gRtbc8TyQnA5+XtAZYQzYyYUTEdk2NzMysilI9iFtttYY8LyQOH4hArH+5aaQNdZMnjPLfdIvI80KiJH1Y0pfT9z0kHVRvPRs8paaRi5avJtjQNPKmWYsGOzQzG4Ly1JFcArwZ+Kf0/UXgO02LyBpWq2mkmVl/y1NHcnBEHCBpFkBELEu9AVuLctNIMxtIeZ5I1koaRjbOOpJGAi83NSprSLUmkG4aaWbNkCcjuQi4EdhF0nnAb4GvNjUqa4ibRprZQMrTausqSTOBd5I1/Z0cEY81PTLbZG4aaWYDSfX6YpS0Z6X0iHiqKRE1WVdXV3R3dw92GGZmhSJpZkR0VZqXp7L9VrL6EZH1tzUWmAfs328RmjXA78yYDa48RVuvL/8u6QDgE02LyKwP3J242eDLU9neQ0Q8CLypCbGY9ZnfmTEbfHnebD+t7PM5ST8BluRY7/uSnpP0SFnajpLulPSn9HOHsnlnSpovaZ6kiWXpB0qak+ZdJEkpfUtJ16b0GZLG9PXgrfj8zozZ4MvzRDK87LMlWZ3JETnWuwKY1CvtDOCuiNgHuCt9R9J+wBSyepdJwCXp3RWAS4GpwD7pU9rmicCyiNgbuBD4eo6YbIjxOzNmgy9PHck5m7LhiLi3wlPCEcBhafqHwHTgCyn9moh4CXhC0nzgIEkLgO0i4j4ASVcCk4Hb0jpnp21dB1wsSR4SuL2cPnFcjzoSGPx3Zlz5b+2mbkYi6ZZa8yPiA33Y364RsTitt1jSLil9FHB/2XILU9raNN07vbTO02lb6yStAHYCnu9DPFZwrfbOjCv/rR3laf77BPAq4Mfp+7HAAuD2foxDFdKiRnqtdTbeuDSVrHiMPfes+FrMkNGOd8Ot1J24xxK3dpQnI5kQEW8v+/4zSfdGxBc3YX/PStotPY3sBjyX0hcCe5QtNxp4JqWPrpBevs5CSZsD2wNLK+00Ii4DLoPshcRNiLsQfDc8+Fz5b62o2TeYeSrbR0p6TemLpLHAyE3c3y3ACWn6BODmsvQpqSXWWLJK9QdSMdhKSYek1lrH91qntK2jgLvbvX7ETWEHnyv/rdUMxPhEeTKSfwWmS5ouaTpwD3BKvZUkXQ3cB4yTtFDSicD5wN9L+hPw9+k7ETEXmAY8CvwSODkiSlfEk4DvAfOBP5NVtANcDuyUKuZPI7UAa2e+Gx587jDTWs1A3GDmabX1S0n7APumpMdT66p66x1bZdY7qyx/HnBehfRu4HUV0v8GHF0vjnay+4hOFlXINHw3PHBarfLfbCBuMPPUkZAyjof6ba/WFK3YFLYdtVLlv9lA3GD2uYsUa12TJ4ziax96PaNGdCJg1IhOvvah1/uiZtbGBqK4NdcTiRWH74bNrNxAFLfmeSFRwHHAayLi3DQ+yasi4oF+i8LMzJqm2TeYeYq2LgHeTPYiIsBK4DtNi8jMzAolT9HWwRFxgKRZABGxTNIWTY7LzMwKIs8TydrUE28ASBoJvNzUqMzMrDDyZCQXATcCu0g6D/gt8NWmRmVmZoWR54XEqyTNJHuRUMDkiHis6ZGZmVkhVM1IJG0XES9I2pGsc8Wry+btGBEVO0g0M7P2UuuJ5CfA+4CZ9OyeXen7ayqtZGabph2HALChoWpGEhHvSz/HDlw4NpT4wpifhwCwIqtb2S7pg5K2L/s+QtLkpkY1RN00axGHnn83Y8+4lUPPv7tfu3FuNQPRdfVQ4iEArMjytNo6KyJWlL5ExHLgrKZFNES124XVF8a+8RAAVmR5MpJKy7iPrj5qtwurL4x94wGxrMjyZCTdkr4paS9Jr5F0IVkFvPVBu11YfWHsGw+IZSVFLALPk5F8GlgDXAv8FPgbcHIzgxqK2u3COtQvjP39z+4hAAyKWwSudhvmvKurK7q7u/t1m3laJ/VulQPZhXUoXyyK2mqrXtzteC5tYBx6/t0VB6EaNaKT353xjkGIaANJMyOiq9K8Wi8kfisiTpX0M3q+RwJARHygH2MsrLzNNttxCNYijo2S53zWqu8q2vFaaylqEXitSvMr08//GIhAiqovF5UiXljbTZ7zWdR/dmt9AzEsbjPUqiO5IP18T0T8uvdnIIIrAl9UhpY857Pd6rts4BS1brFWRrKbpP8FfEDSBEkHlH8GKsBW54vK0JLnfBb1n91aX1EbXdQq2voKcAYwGvgGWR9bJQEMbs1Pizh94riKFa++qBRTnvPZjvVdNnCKWARes9VWGq/9yxFx7sCF1FyD1WrLisPn02xjtVpt1W3+m1Y+sCmRDYJmZCRmZkNdrYwkzwuJ90t6Uz8HtEDSHEmzJXWntB0l3SnpT+nnDmXLnylpvqR5kiaWpR+YtjNf0kXpCcrMzAZQnozkcLLM5M+SHk4X7of7Yd+HR8T4shzuDOCuiNgHuCt9R9J+wBRgf2AScEkaQx7gUmAqsE/6TOqHuMzMrA/ydL74D02PInMEcFia/iEwHfhCSr8mIl4CnpA0HzhI0gJgu4i4D0DSlcBk4LYBitfMzMjxRBIRTwJ7AO9I06vyrFdvs8AdkmZKmprSdo2IxWmfi4FdUvoo4OmydRemtFFpune6mZkNoLpPJJLOArqAccAPgA7gx8ChDez30Ih4RtIuwJ2SHq8VQoW0qJG+8QayzGoqwJ577tnXWM3MrIY8TxYfBD4A/BUgIp4Bhjey07QNIuI54EbgIOBZSbsBpJ/PpcUXkj0RlYwGnknpoyukV9rfZRHRFRFdI0eObCR0MzPrJU9GsiayNsIBIGmbRnYoaRtJw8u29W7gEeAW4IS02AnAzWn6FmCKpC0ljSWrVH8gFX+tlHRIaq11fNk6ZpZDEce+sNaTp7J9mqT/AkZI+jjwMeC7DexzV+DG1FJ3c+AnEfFLSX9I+zoReAo4GiAi5kqaBjwKrANOjojSa8cnAVcAnWSV7K5oN8spb8/VZvXkGo9E0t+TPTkA3BERdzY1qibyC4lmmVYe+8JazyaNR9LLHLK7/kjTZlZw7rna+kueVlv/QtaB491kLaW+LenciPh+s4Mrqlbuq6mVY7OBVdSxL6z15KlsPx2YEBEfjYgTgAPJXhS0Clp5zOVWjs0GnrvDt/6SJyNZCKws+76Sni8IWplaI+wNtlaOzQZeUce+sNaTp45kETBD0s1kdSRHAA9IOg0gIr7ZxPhaRt4ioWrly4uWr2bsGbcOanGSy8SttyKOfWGtJ09G8uf0KSm9q9HQS4lFkqeZZCmjqdUGLqqs23tfzarDcJm4mTVD3YwkIs4ZiEBaWa0iockTRm2U0dRTvm65Zrfr92iOZtYMeVpt3UOFPqwiom0amtcrEqqU0WzKNuvVYTT6pJJniFi36uo//l1au8hTtPW5sumtgCPJ3jBvG9WKhILqL3WViMo9SVYqTqpVv9JfTyq1ysT9pnP/8e/S2kmebuRnln1+FxGnAQcPQGwto1IzyZJFy1dX7IYYYJhUMRNR2mZv1eoqhkmb1Nqqr/0otUqrrqHQ/1Or/C7NBkKeoq0dy75uRvYeyauaFlELKi8SqvZk0vvJo7NjWNXirkjbKt/2TbMWsWrNxg96tbZTq7VVrTvi0rH0LnJphVZdQ+VOvhV+l2YDJU/R1kw2XCvXAU8AJzYzqFZUKhIac8atFecHWTv88otztYwHNr6wV6qsH9HZwdkf2L/qdmq1tqp2R3z2LXN5ad3LFS/UrdCqq17DhqJohd+l2UDJ02pr7EAEUgT1ilhKd/alStZSsVe1JsGr167nnJ/NZestNq/41LHNlpu/cvGs1Nrq8H1Hcuj5d1eszK1257t89dqKcZx67Wx22LqDjs3E2pc3RNzMVl2VKqOHyp28W8hZO6makUh6E/B0RPx3+n48WUX7k8DZEbF0YEJsDcd99z5+9+fah3z6Tx+i+8mlXD9z0SsXkErFXuWWrVrLslUbX9xhw8WzUmurw/cd2WM/vYuAqt0R17Js1Vo6hokRnR2sWL22Ty2N+tpCqVoR1oitOyr+Pop2J5+nhZzZUFG1G3lJDwLvioilkt4OXAN8GhgP/F1EHDVgUfajTelG/ks3zeHH9z+Va9lhEutzdM2fZ50RnR1ss+XmFS9E48+5o+LTBWRFbL0zGsjuiLfq2KxqxlW+fl+6Ea/0Hk1nx7Ca3W1Ua+02orOjR9Fbnm0VkZsGW9HU6ka+VqutYWVPHccAl0XE9RHxZWDv/g6ylV09I3/XYn3NRErr9G751bGZ+OuadRU7WLxp1qKqmQhp2etnLuLIA0dt1I/SWe/fv2oLtJK+FiNVq9c49drZjD/njopFgtX2sWL12iHf/5M7z7ShplYdyTBJm0fEOuCdwNSc6w05m5I59FV5EdioEZ2sWrNuoyeH1WvXc9q02Wzf2VF3e6vXrueex5dUfbKo1RCgUjFSrTvoWhnP8tVrOW3abKBnq6taldFDvf+nodKgwKykVoZwNfBrSc8Dq4HfAEjaG1gxALG1nVIm8rsz3sHYKq3DXg7qFk2VlHcUefi+I7nn8SU9MgKoXInfu0K4XpPcevUxLwec87O5PS6S7VwZPVQaFOThIrz2ULVoKyLOAz5LNib6W2NDZcpmZHUl1gSli0l/VS6Xik5+fP9TGxWlALmKkeq9XFfrhc2S3plfO3dhXu3cFq1BQT0uwmsfNYuoIuL+Cml/bF44NmLrrNjq9InjOPXa2U3bTykj+N0Z76h78a53B11a/7PTHqpZDHjo+Xf3uCPtryKsot31tsvTmIvw2kdb1XUUwbJVa5lw7h0sX7W2ZrPhEZ0dSD3v9DdTVoyUV96ilDwv102eMIruJ5fWbN3WjLfUi/gmfLs0DW6nIrx254ykBeWpA1m1Zh29b/633HwYRx44KndT5TxFKbW6bim/g75p1iKun1m/yKL0EmZ/XUSLetc71BsUgN/ubyd5htq1FrRmffR4Ax2yC2jepsp5ilJKd/u9M7YRnR0b1Wf0pSv9ZavW9lu5ue96W5fHhG8fzkiGmDxNlYdJuSq2q2UO5V23lDRy4W6kV9x2qbguonZuUNFuXLTVZjo2Excc/cZc/8x9udvflC5Z8uyrnnapuC6qdijCM2ckbWfbrTZ+mqimVhl375ZSlbpkqaQvA33l0S4V12atrPAZiaRJwP8DhgHfi4jzBzmkASWxUaV7LctXrc3dXLba3f7h+47cqKVUqUuW0kuP1UIKNh5jpdEnCN/1mg2uQteRSBoGfAf4B2A/4FhJ+/XnPlr95am+9t4yYuuO3C+JVSvjvufxJRVbSpW6ZHni/PcyqsoTRmkbLjc3GzqK/kRyEDA/Iv4CIOka4Ajg0f7aQZGHRq00amMEfWouW+lu/1+rvChZXs9Rq+7CTxBmQ0uhn0iAUUB5e9eFKa0HSVMldUvqXrJkSZ92UNRmpJ0dwzjukD03uvNfUaXX4L4cZ56WUm6xY9Y+iv5E0rv3dahQlxsRlwGXQTYeSV920GhrpIGyw9YdbL1F5bFLym3KsL295W0p5ScPs/ZQ9IxkIbBH2ffRwDP9uYNm93lVboetO3jvG3ar2fqpYzOBYO36nsPhnvX+/XNdtPujuaxbSplZuaJnJH8A9pE0FlgETAH+qT93ULo4NiMzGSZx7MF78O+TX98jvevVO75ykR6xdQcR9Bj6Fjb9It5fmYCfNsyspOpQu0Uh6T3At8ia/34/dX9f1aYMtWtm1u5qDbVb9CcSIuIXwC8GOw4zs3ZV9FZbZmY2yJyRmJlZQ5yRmJlZQ5yRmJlZQwrfaquvJC0BnmxgEzsDz/dTOINlKBwDDI3jGArHAD6OVtKsY3h1RIysNKPtMpJGSequ1gSuKIbCMcDQOI6hcAzg42glg3EMLtoyM7OGOCMxM7OGOCPpu8sGO4B+MBSOAYbGcQyFYwAfRysZ8GNwHYmZmTXETyRmZtYQZyQVSJokaZ6k+ZLOqDBfki5K8x+WdMBgxFlPjuM4TNIKSbPT5yuDEWctkr4v6TlJj1SZX5RzUe84inAu9pB0j6THJM2VdEqFZVr6fOQ8hiKci60kPSDpoXQc51RYZuDORUT4U/Yh60X4z8BrgC2Ah4D9ei3zHuA2soG1DgFmDHbcm3gchwE/H+xY6xzH24EDgEeqzG/5c5HzOIpwLnYDDkjTw4E/Fu1/I+cxFOFcCNg2TXcAM4BDButc+IlkY6+MAx8Ra4DSOPDljgCujMz9wAhJuw10oHXkOY6WFxH3AktrLFKEc5HnOFpeRCyOiAfT9ErgMTYe2rqlz0fOY2h56ff7YvrakT69K7wH7Fw4I9lYnnHgc40VP8jyxvjm9Hh8m6T9Bya0flWEc5FXYc6FpDHABLI74XKFOR81jgEKcC4kDZM0G3gOuDMiBu1cFH48kibIMw58rrHiB1meGB8k6/bgxTRA2E3APs0OrJ8V4VzkUZhzIWlb4Hrg1Ih4offsCqu03PmocwyFOBcRsR4YL2kEcKOk10VEeR3cgJ0LP5FsLM848E0fK74f1I0xIl4oPR5HNkBYh6SdBy7EflGEc1FXUc6FpA6yC/BVEXFDhUVa/nzUO4ainIuSiFgOTAcm9Zo1YOfCGcnGXhkHXtIWZOPA39JrmVuA41OriEOAFRGxeKADraPucUh6lSSl6YPI/h7+Z8AjbUwRzkVdRTgXKb7Lgcci4ptVFmvp85HnGApyLkamJxEkdQLvAh7vtdiAnQsXbfUSEeskfQq4nQ3jwM+V9Mk0/z/JhvZ9DzAfWAX882DFW03O4zgKOEnSOmA1MCVSc49WIelqslY0O0taCJxFVrFYmHMBuY6j5c8FcCjwEWBOKpsH+CKwJxTmfOQ5hiKci92AH0oaRpbRTYuInw/WdcpvtpuZWUNctGVmZg1xRmJmZg1xRmJmZg1xRmJmZg1xRmJmNoSpToehvZa9sKyzyj9KWp5rH261Zf1B0ifImiAuG+xYzGwDSW8HXiTrd+t1fVjv08CEiPhYvWX9RGK5SFqf7lIekfRTSVuXzfsKsDRvJiJpuqSuNP2L0otVvZY5W9Ln+iv+IpD0Yv2lWkO1WCX9fqBjSfs9TNLP0/QHVGHYhHZVqcNQSXtJ+qWkmZJ+I2nfCqseC1ydZx/OSCyv1RExPt3RrAE+WZoREedGxE8rrSSp5kuvEfGe1MXDoEkvdbW9eucqj4h4S3/E0mAMt0TE+YMdR4u7DPh0RBwIfA64pHympFcDY4G782zMGYltit8Ae0vaJpW//kHSLElHAEj6aHpq+Rlwh6ROSdcoG1znWqCztCFJC0r9GEn6P8oG4voVMK5smY+nfTwk6frS05Cko9MT0kOS7u0dZLpLvVfSjZIelfSfkjZL816UdK6kGWQ9vZ6WtvWIpFPTMl+X9L/Ltne2pM+mLicuSMvOkXRM2TKfT2kPSTq/TvxjJd2X5v1b2Ta2lXSXpAfTtkq/120k3Zq280j5fsvWnS7pW5J+n5Y5KKXvKOmmdA7ul/SGsmO6TNIdwJUVtnd6iu9hVRg8qcLyL5b97qdLuk7S45Kukl7pduRNKb6HlA3ONFxZT7YXlO3rEzm2Myml/Rb4UFkMH5V0cZp+dfpdPpx+7lnvGIY6ZR1WvgX4qbK3+/+L7E35clOA61LHkPU1a6ATf4bWB3gx/dwcuBk4Cfgq8OGUPoJskKBtgI+SdRi3Y5p3GlkXLQBvANYBXen7AmBn4EBgDrA1sB1Ztw6fS8vsVBbHv5PdSZGWH1Xaf4WYDwP+Rja41zDgTuCoNC+Af0zTpX1vA2wLzCXrXnwC8Ouy7T1K1pXGkWlbw4BdgafSP+I/AL8Htk7L71gn/luA49P0yb1+x9ul6Z3T70Jpv98t29b2FY55emkZssG0HknT3wbOStPvAGan6bOBmUBnhW29m+zOVWQ3nT8H3l7+91Dj7+QwYAVZR4GbAfcBbyUbZO0vwJvSctul450KfCmlbQl0k90RV9vOVmRdpO+T4ptGGoyK7O/v4jT9M+CENP0x4KbB/l8apP/fMWV/C9sBi+ssPwt4S97t+4nE8upMdy/dZBfOy8kuNGek9Olk/9ylO747I6JULvt24McAEfEw8HCF7b8NuDEiVkXWrXd5B5OvU1aOOwc4DiiND/E74ApJHye7qFfyQGSDe60nK+99a0pfT9YDLCntxoj4a2S9vt4AvC0iZgG7SNpd0huBZRHxVFr+6ohYHxHPAr8G3kTWcd4PImJVOtbS8VeL/1A2lEH/qCxmAV+V9DDwK7IxJHYly+zelZ6U3hYRK6oc89Vp//cC2ymrg3praR8RcTewk6Tt0/K3RMTqCtt5d/rMIutafV/61p36AxGxMCJeBmaTXczGkV3E/pBieSEi1qX9HJ/+lmYAO5Xtq9J29gWeiIg/RXbl+3GVGN4M/CRN/4gN579tpf+vJyQdDa8MyfvG0nxJ44AdyDLtXNxpo+W1OiLGlyekIoYjI2Jer/SDgb/2Wj9P88Bqy1wBTI6IhyR9lOwulYj4ZNrXe4HZksZHRO9eWntvs/T9b7Hhsb3SuA0l15F14vcqslEmay2vKsdQMf4q8UGW2YwEDoyItZIWAFtFxB8lHUjWEd/XJN0REedWWL/SMdcam6L3uSoR8LWI+K8q8+t5qWx6Pdn1ptrvSGRParf3SJQOq7IdqmynnrZrpqrKHYYeB1wq6UtknYdeQzYcN2SV7NekDDoXP5FYI24HPl1WZj2hynL3kv3hIul1ZMVblZb5oLL6lOHA+8vmDQcWKxtH4rhSoqS9ImJGRHwFeJ6eYy+UHJTqIjYDjgF+W2XfkyVtLWkb4INk9UCQ/YNNIctMritb/phUrj+S7InrAeAO4GNldSA71oqf7IlqSpouT98eeC5lIocDr07b2x1YFRE/Bv6DbAz4So5Jy7+VrOvwFfQ8B4cBz8fGAzr1dns6nm3TeqMk7VJnnXoeB3aX9Ka0zeHKKvlvJ+txtyOlvzadi1rbGStpr/T92CrL/Z6ev+NK539Ii4hjI2K3iOiIiNERcXlEPBERkyLijRGxX/kNSUScHRF9avXmJxJrxL8B3wIeTpnJAuB9FZa7FPhBKqqZTXbR7SEiHlRWET8beJINF3KAL5MVdzxJVrwzPKVfIKlURn4XG+6oyt0HnA+8nuxiemOVfV9RFtf3UrEWkXW9PxxYFBvGcriRrMjkIbI73M9HxH8Dv5Q0HuiWtIasG+8v1oj/FOAnkk5hQzEbwFXAzyR1p99HaZyJ16djfhlYS1ZPVckyZc1wtyOrF4CsLqR0DlYBJ1RZt/z3coekvwPuS/cKLwIfJhvadZNExBpljQS+rWwcjdVkRYLfIyuyejD9LS0BJtfYzt8kTQVulfQ8WQZR6R2JzwDfl3R62mardWs/JPiFRBuy0p335yKiUuY2JEmaTnbM3YMdi7UPF22ZmVlD/ERiZmYN8ROJmZk1xBmJmZk1xBmJmZk1xBmJmZk1xBmJmZk1xBmJmZk15P8D4NO9YhAy2rIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(df.perdidas, df.superficie)\n",
    "plt.xlabel('Pérdidas provocadas por el incendio')\n",
    "plt.ylabel('Superficie quemada en el incendio')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "f0ade7b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.sort_values(\"superficie\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "71157003",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.reset_index(drop = True, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "318ea67a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.perdidas = df.perdidas.fillna(method = \"ffill\").fillna(method = \"bfill\") "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "4742ef7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 82640 entries, 0 to 82639\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   superficie   82640 non-null  float64\n",
      " 1   idcomunidad  82640 non-null  object \n",
      " 2   idprovincia  82640 non-null  object \n",
      " 3   idmunicipio  82640 non-null  object \n",
      " 4   causa        82640 non-null  object \n",
      " 5   muertos      82640 non-null  int64  \n",
      " 6   heridos      82640 non-null  int64  \n",
      " 7   time_ctrl    82640 non-null  int64  \n",
      " 8   time_ext     82640 non-null  int64  \n",
      " 9   personal     82640 non-null  int64  \n",
      " 10  medios       82640 non-null  int64  \n",
      " 11  perdidas     82640 non-null  float64\n",
      "dtypes: float64(2), int64(6), object(4)\n",
      "memory usage: 7.6+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b3eece0",
   "metadata": {},
   "source": [
    "Información variables numéricas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "8ba16dfb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>superficie</th>\n",
       "      <th>muertos</th>\n",
       "      <th>heridos</th>\n",
       "      <th>time_ctrl</th>\n",
       "      <th>time_ext</th>\n",
       "      <th>personal</th>\n",
       "      <th>medios</th>\n",
       "      <th>perdidas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>82640.000000</td>\n",
       "      <td>8.264000e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>19.888085</td>\n",
       "      <td>0.000702</td>\n",
       "      <td>0.007357</td>\n",
       "      <td>236.247302</td>\n",
       "      <td>523.010733</td>\n",
       "      <td>21.188093</td>\n",
       "      <td>3.143224</td>\n",
       "      <td>2.663368e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>223.787536</td>\n",
       "      <td>0.054333</td>\n",
       "      <td>0.143154</td>\n",
       "      <td>901.186810</td>\n",
       "      <td>2560.985781</td>\n",
       "      <td>48.210975</td>\n",
       "      <td>6.180303</td>\n",
       "      <td>3.988815e+05</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-2.896100e+04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>134.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>9.000000e+01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>118.000000</td>\n",
       "      <td>221.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.250000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.150000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>210.000000</td>\n",
       "      <td>415.000000</td>\n",
       "      <td>23.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>5.160250e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>28879.100000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>132555.000000</td>\n",
       "      <td>529682.000000</td>\n",
       "      <td>3979.000000</td>\n",
       "      <td>310.000000</td>\n",
       "      <td>3.064011e+07</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         superficie       muertos       heridos      time_ctrl       time_ext  \\\n",
       "count  82640.000000  82640.000000  82640.000000   82640.000000   82640.000000   \n",
       "mean      19.888085      0.000702      0.007357     236.247302     523.010733   \n",
       "std      223.787536      0.054333      0.143154     901.186810    2560.985781   \n",
       "min        1.000000      0.000000      0.000000       0.000000       0.000000   \n",
       "25%        1.500000      0.000000      0.000000      64.000000     134.000000   \n",
       "50%        3.000000      0.000000      0.000000     118.000000     221.000000   \n",
       "75%        7.150000      0.000000      0.000000     210.000000     415.000000   \n",
       "max    28879.100000     11.000000     12.000000  132555.000000  529682.000000   \n",
       "\n",
       "           personal        medios      perdidas  \n",
       "count  82640.000000  82640.000000  8.264000e+04  \n",
       "mean      21.188093      3.143224  2.663368e+04  \n",
       "std       48.210975      6.180303  3.988815e+05  \n",
       "min        0.000000      0.000000 -2.896100e+04  \n",
       "25%        5.000000      1.000000  9.000000e+01  \n",
       "50%       11.000000      2.000000  1.250000e+03  \n",
       "75%       23.000000      3.000000  5.160250e+03  \n",
       "max     3979.000000    310.000000  3.064011e+07  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a50bf11",
   "metadata": {},
   "source": [
    "Tratamiento registros con pérdidas negativas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f2c35e6d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df[df[\"perdidas\"] < 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "614e07d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[df[\"perdidas\"] >= 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3318f0fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.reset_index(drop = True, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e78fed2",
   "metadata": {},
   "source": [
    "Análisis variables numéricas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "07c75f63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 936x1440 with 8 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(4, 2, figsize = (13, 20))\n",
    "ax[0, 0].hist(df.superficie, bins = 20)\n",
    "ax[0, 0].set_xlabel(\"Superficie quemada\")\n",
    "ax[0, 0].set_ylabel(\"Frecuencia absoluta\")\n",
    "\n",
    "ax[2, 1].hist(df.muertos, bins = 20)\n",
    "ax[2, 1].set_xlabel(\"Muertos\")\n",
    "ax[2, 1].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[1, 0].hist(df.heridos, bins = 20)\n",
    "ax[1, 0].set_xlabel(\"Heridos\")\n",
    "ax[1, 0].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[1, 1].hist(df.time_ctrl, bins = 20)\n",
    "ax[1, 1].set_xlabel(\"Tiempo control incendio\")\n",
    "ax[1, 1].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[0, 1].hist(df.time_ext, bins = 20)\n",
    "ax[0, 1].set_xlabel(\"Tiempo extinción incendio\")\n",
    "ax[0, 1].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[2, 0].hist(df.personal, bins = 20)\n",
    "ax[2, 0].set_xlabel(\"Personal\")\n",
    "ax[2, 0].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[3, 0].hist(df.medios, bins = 20)\n",
    "ax[3, 0].set_xlabel(\"Medios\")\n",
    "ax[3, 0].set_ylabel(\"Frecuencia absoluta\")\n",
    "ax[3, 1].hist(df.perdidas, bins = 20)\n",
    "ax[3, 1].set_xlabel(\"Pérdidas\")\n",
    "ax[3, 1].set_ylabel(\"Frecuencia absoluta\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e121ae2a",
   "metadata": {},
   "source": [
    "Definición variable objetivo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "bb256d99",
   "metadata": {},
   "outputs": [],
   "source": [
    "target = np.zeros(len(df))\n",
    "for i in range(0, len(df)):\n",
    "    if (df.muertos[i] > 0 or df.heridos[i] > 0):\n",
    "        target[i] = 1\n",
    "    elif df.perdidas[i] > 5000:\n",
    "        target[i] = 1\n",
    "    else:\n",
    "        target[i] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "644bb2e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop([\"muertos\", \"heridos\", \"perdidas\"], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3832df86",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"target\"] = np.zeros(len(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "51df5a04",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"target\"] = target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "74b85389",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"target\"] = df[\"target\"].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "420b0609",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 82638 entries, 0 to 82637\n",
      "Data columns (total 10 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   superficie   82638 non-null  float64\n",
      " 1   idcomunidad  82638 non-null  object \n",
      " 2   idprovincia  82638 non-null  object \n",
      " 3   idmunicipio  82638 non-null  object \n",
      " 4   causa        82638 non-null  object \n",
      " 5   time_ctrl    82638 non-null  int64  \n",
      " 6   time_ext     82638 non-null  int64  \n",
      " 7   personal     82638 non-null  int64  \n",
      " 8   medios       82638 non-null  int64  \n",
      " 9   target       82638 non-null  int64  \n",
      "dtypes: float64(1), int64(5), object(4)\n",
      "memory usage: 6.3+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6ab9d26",
   "metadata": {},
   "source": [
    "Diagramas de cajas relación \"target\" y variables numéricas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e6a1b727",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data = df, x = \"target\", y = \"superficie\", kind = \"box\")\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "5bf23996",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data = df, x = \"target\", y = \"time_ctrl\", kind = \"box\")\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "712984fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data = df, x = \"target\", y = \"time_ext\", kind = \"box\")\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "31184030",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAW4AAAFuCAYAAAChovKPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAfJ0lEQVR4nO3df5DcdZ3n8ecrMxh+KAcMA8ZJXNBE9vihUaYitwoGmCwj3gme612oPTO1BTcuokF2665gi6rVK7G8LX+scMIRI0XnbjWVu9UjUGG8JEcO8FAYFBMCIqNECQnJ0BZrACvszLzvj/5Otgmdpifpb3/n0/16VHX19/Pp77f7PcXMi08+329/vooIzMwsHXOKLsDMzGbGwW1mlhgHt5lZYhzcZmaJcXCbmSWmu+gC8jI4OBgjIyNFl2FmdiRUq7NtR9wvvPBC0SWYmeWibYPbzKxdObjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMbkHt6QuST+VdE/WPknSRklPZ88nVu17g6QxSU9JuqSq/1xJ27LXbpZU89tEZmadoBUj7muBJ6va1wObI2IRsDlrI+lMYDlwFjAI3CqpKzvmNmAYWJQ9BltQt5k1SblcZuXKlZTL5aJLaQu5Brek+cBHgNVV3ZcBpWy7BFxe1b82IvZHxDPAGLBE0jzg+Ih4KCq361lTdYyZJaBUKrFt2zbWrFlTdCltIe8R998C/xGYquo7NSJ2A2TPp2T9fcCzVfvtzPr6su2D+19H0rCkUUmj4+PjTfkBzOzIlMtlRkZGiAhGRkY86m6C3IJb0r8E9kbEo40eUqMv6vS/vjNiVUT0R0R/b29vgx9rZnkqlUpMTVXGbpOTkx51N0GeI+4PAB+VtANYC1wk6b8De7LpD7Lnvdn+O4EFVcfPB3Zl/fNr9JtZAjZt2sTExAQAExMTbNy4seCK0pdbcEfEDRExPyJOo3LS8f9ExL8D1gND2W5DwF3Z9npguaS5kk6nchLy4Ww6ZZ+k87KrSVZUHWNms9zAwADd3ZWl/7u7u1m2bFnBFaWviOu4vwwsk/Q0sCxrExHbgXXAE8AIcE1ETGbHXE3lBOcY8Evg3lYXbWaHZ2hoiDlzKlHT1dXFihUrCq4ofS25A05EbAG2ZNtl4OJD7HcTcFON/lHg7PwqNLO89PT0MDg4yN13383g4CA9PT1Fl5S8tr11mZnNHkNDQ+zYscOj7SZR5dLo9tPf3x+jo6NFl2FmdiQ6656TZmbtysFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpaY3IJb0tGSHpb0M0nbJX0h6/+8pOckPZY9Lq065gZJY5KeknRJVf+5krZlr90sSXnVbWY223Xn+N77gYsi4iVJRwEPSro3e+3rEfGV6p0lnQksB84C3gZskvSuiJgEbgOGgR8BG4BB4F7MzDpQbiPuqHgpax6VPaLOIZcBayNif0Q8A4wBSyTNA46PiIciIoA1wOV51W1mNtvlOsctqUvSY8BeYGNE/Dh76TOStkq6Q9KJWV8f8GzV4Tuzvr5s++D+Wp83LGlU0uj4+HgzfxQzs1kj1+COiMmIWAzMpzJ6PpvKtMc7gcXAbuCr2e615q2jTn+tz1sVEf0R0d/b23uE1ZuZzU4tuaokIl4EtgCDEbEnC/Qp4FvAkmy3ncCCqsPmA7uy/vk1+s0sEeVymZUrV1Iul4supS3keVVJr6QTsu1jgAHg59mc9bSPAY9n2+uB5ZLmSjodWAQ8HBG7gX2SzsuuJlkB3JVX3WbWfKVSiW3btrFmzZqiS2kLeY645wH3SdoKPEJljvse4G+yS/u2AhcC1wFExHZgHfAEMAJck11RAnA1sJrKCctf4itKzJJRLpcZGRkhIhgZGfGouwlyuxwwIrYC763R/8k6x9wE3FSjfxQ4u6kFmllLlEolpqamAJicnGTNmjVcd911BVeVNn9z0sxytWnTJiYmJgCYmJhg48aNBVeUPge3meVqYGCA7u7KP+67u7tZtmxZwRWlz8FtZrkaGhpizpxK1HR1dbFixYqCK0qfg9vMctXT08Pg4CCSGBwcpKenp+iSkpfnWiVmZkBl1L1jxw6PtptEleU/2k9/f3+Mjo4WXYaZ2ZGouRKqp0rMzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS0xuwS3paEkPS/qZpO2SvpD1nyRpo6Sns+cTq465QdKYpKckXVLVf66kbdlrN0uqect6M7NOkOeIez9wUUS8B1gMDEo6D7ge2BwRi4DNWRtJZwLLgbOAQeBWSV3Ze90GDAOLssdgjnWbmc1quQV3VLyUNY/KHgFcBpSy/hJwebZ9GbA2IvZHxDPAGLBE0jzg+Ih4KCICWFN1jJlZx8l1jltSl6THgL3Axoj4MXBqROwGyJ5PyXbvA56tOnxn1teXbR/cX+vzhiWNShodHx9v6s9iZjZb5BrcETEZEYuB+VRGz2fX2b3WvHXU6a/1easioj8i+nt7e2dcr5lZClpyVUlEvAhsoTI3vSeb/iB73pvtthNYUHXYfGBX1j+/Rr+ZWUfK86qSXkknZNvHAAPAz4H1wFC22xBwV7a9Hlguaa6k06mchHw4m07ZJ+m87GqSFVXHmJl1nO4c33seUMquDJkDrIuIeyQ9BKyTdCXwG+ATABGxXdI64AlgArgmIiaz97oauBM4Brg3e5iZdSRVLtRoP/39/TE6Olp0GWZmR6Lmd1b8zUkzs8Q4uM3MEuPgNjNLjIPbzCwxDm4zy93Y2Bgf+chHGBsbK7qUtuDgNrPcffGLX+Tll1/mi1/8YtGltAUHt5nlamxsjB07dgCwY8cOj7qbwMFtZrk6eJTtUfeRc3CbWa6mR9uHatvMObjNLFennXZa3bbNnIPbzHJ144031m3bzDm4zSxXCxcuPDDKPu2001i4cGGxBbUBB7eZ5e7GG2/kuOOO82i7Sbw6oJnZ7OXVAc3M2oGD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS4yD28wsMQ5uM7PEOLjNzBLj4DYzS0xuwS1pgaT7JD0pabuka7P+z0t6TtJj2ePSqmNukDQm6SlJl1T1nytpW/bazZJqrphlZtYJunN87wngLyPiJ5LeAjwqaWP22tcj4ivVO0s6E1gOnAW8Ddgk6V0RMQncBgwDPwI2AIPAvTnWbmY2a+U24o6I3RHxk2x7H/Ak0FfnkMuAtRGxPyKeAcaAJZLmAcdHxENRWTx8DXB5XnWbmc12LZnjlnQa8F7gx1nXZyRtlXSHpBOzvj7g2arDdmZ9fdn2wf1mZh0p9+CW9Gbg74HPRcTvqEx7vBNYDOwGvjq9a43Do05/rc8aljQqaXR8fPxISzczm5VyDW5JR1EJ7b+LiO8BRMSeiJiMiCngW8CSbPedwIKqw+cDu7L++TX6XyciVkVEf0T09/b2NveHMTObJfK8qkTAt4EnI+JrVf3zqnb7GPB4tr0eWC5prqTTgUXAwxGxG9gn6bzsPVcAd+VVt5nZbJfnVSUfAD4JbJP0WNb3V8AVkhZTme7YAXwKICK2S1oHPEHlipRrsitKAK4G7gSOoXI1ia8oMbOO5bu8m5nNXr7Lu5lZO3Bwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiXFwm5klxsFtZpYYB7eZWWIc3GZmiam7rKuk99V7ffqekmZm1jpvtB73V+u8FsBFTazFzMwaUDe4I+LCVhViZmaNafgOOJLOBs4Ejp7ui4g1eRRlZmaH1lBwS/prYCmV4N4AfBh4EHBwm5m1WKNXlfwJcDHwfET8GfAeYG5uVZmZ2SE1Gty/j4gpYELS8cBe4B35lWVmZofS6Bz3qKQTgG8BjwIvAQ/nVZSZmR1aQ8EdEZ/ONv+rpBHg+IjYml9ZZmZ2KDO5qqQP+IPpYyRdEBH351WYmZnV1uhVJf8Z+LfAE8Bk1h2Ag9vMrMUaHXFfDpwREftzrMXMzBrQ6FUlvwKOyrMQMzNrTKMj7leAxyRtBg6MuiNiZS5VmZnZITUa3Ouzh5mZFayhqZKIKAHfpXIN96PAd7K+Q5K0QNJ9kp6UtF3StVn/SZI2Sno6ez6x6pgbJI1JekrSJVX950ralr12syQdzg9rZtYOGgpuSUuBp4FvArcCv5B0wRscNgH8ZUT8c+A84BpJZwLXA5sjYhGwOWuTvbYcOAsYBG6V1JW9123AMLAoeww2+POZmbWdRk9OfhX444j4UERcAFwCfL3eARGxe/pGCxGxD3gS6AMuA6ZH6yUqV6yQ9a+NiP0R8QwwBiyRNI/KF34eioigsrDV5ZiZdahGg/uoiHhquhERv2AGV5lIOg14L/Bj4NSI2J29z27glGy3PuDZqsN2Zn192fbB/bU+Z1jSqKTR8fHxRsszM0tKo8E9KunbkpZmj9VU5rrfkKQ3A38PfC4ifldv1xp9Uaf/9Z0RqyKiPyL6e3t7GynPzCw5jQb31cB2YCVwbbb95290kKSjqIT230XE97LuPdn0B9nz3qx/J7Cg6vD5wK6sf36NfjOzjtToVSX7I+JrEfGvgSupnFys+y3K7MqPbwNPRsTXql5aDwxl20PAXVX9yyXNlXQ6lZOQD2fTKfsknZe954qqY8zMOk6ja5VsAT6a7f8YMC7p/0bEX9Q57APAJ4Ftkh7L+v4K+DKwTtKVwG+ATwBExHZJ66ishzIBXBMR0+uiXA3cCRwD3Js9zMw6kioXarzBTtJPI+K9kq4CFkTEX0vaGhHvzr/Ew9Pf3x+jo6NFl2FmdiRqfmel0Tnu7mw++t8A9zStJDPrCOVymZUrV1Iul4supS00GtxfAH4AjEXEI5LeQeULOWZmb6hUKrFt2zbWrPH9xZvhDYM7+/bigoh49/SdcCLiVxHx8dyrM7PklctlRkZGiAhGRkY86m6CNwzu7AThR1tQi5m1oVKpxNTUFACTk5MedTdBo1Ml/0/Sf5F0vqT3TT9yrczM2sKmTZuYmJgAYGJigo0bNxZcUfoaXdb1j7Ln/1TVF8BFzS3HzNrNwMAAGzZsYGJigu7ubpYtW1Z0Sclr9C7vF+ZdiJm1p6GhIUZGRgDo6upixYoVBVeUvkaXdT01W6vk3qx9ZvYFGjOzunp6ehgcHEQSg4OD9PT0FF1S8hqd476TyuWAb8vavwA+l0M9ZtaGhoaGOOecczzabpJGg/vkiFgHTAFExAQwWf8QM7OKnp4ebr75Zo+2m6TR4H5ZUg/ZcqqSzgP+IbeqzMzskBq9quQvqKze9w5JPwR6gT/JrSozMzukRoP7CeD7wCvAPuB/UZnnNjOzFmt0qmQN8IfAl4BbqKyV/d/yKsrM2osXmWquRoP7jIi4KiLuyx7DwLvyLMzM2ocXmWquRoP7p9kJSQAkvR/4YT4lmVk78SJTzddocL+fynolOyTtAB4CPiRpm6StuVVnZsnzIlPN1+jJycFcqzCztlVrkanrrruu4KrS1uhaJb/OuxAza09eZKr5Gp0qMTM7LENDQ8yZU4kaLzLVHA5uM8uVF5lqvkbnuM3MDtvQ0BA7duzwaLtJFBFF15CL/v7+GB0dLboMM7MjoVqdnioxs9z5m5PN5eA2s9zdfvvtbN26lVWrVhVdSltwcJtZrsrlMps2bQJg48aNHnU3gYPbzHJ1++23H/jm5NTUlEfdTeDgNrNcbd68+TXt6dG3Hb7cglvSHZL2Snq8qu/zkp6T9Fj2uLTqtRskjUl6StIlVf3nZmuijEm6WVLNs6xmNjsd/CfrP+Ejl+eI+05qr3Hy9YhYnD02QOWu8cBy4KzsmFsldWX73wYMU1kDfNEh3tPMZqmLL764bttmLrfgjoj7gd82uPtlwNqI2B8RzwBjwBJJ84DjI+KhqFxwvga4PJeCzSwXw8PDB77yPmfOHIaHhwuuKH1FzHF/RtLWbCrlxKyvD3i2ap+dWV9ftn1wf02ShiWNShodHx9vdt1mdhh6enro66v82fb19fkr703Q6uC+DXgnsBjYDXw166816RV1+muKiFUR0R8R/b29vUdYqpk1Q7lc5vnnnwdgz549vhywCVoa3BGxJyImI2IK+BawJHtpJ7Cgatf5wK6sf36NfjNLRKlUYnppjampKd9IoQlaGtzZnPW0jwHTV5ysB5ZLmivpdConIR+OiN3APknnZVeTrADuamXNncBfR7Y81bqRgh2ZPC8H/C6VW5ydIWmnpCuBv6m63dmFwHUAEbEdWAc8AYwA10TEZPZWVwOrqZyw/CVwb141dyrfyNXyNDAwcOASQEm+kUITeHXADlcul7niiit49dVXmTt3Lt/5znd88siaamxsjKuuuupAe/Xq1SxcuLDAipLi1QHt9XwjV8vb+vXrXzPivvvuuwuuKH0O7g7n+UfL26ZNmw6cnIwI/441gYO7ww0MDNDdXbkRkm/kannw71jzObg7nG/kankbGhp6zeqA/h07cg7uDucbuZqlx8FtDA0Ncc4553gkZLkolUqvafsE+JHz5YBmlqtLL72UV1555UD72GOPZcOGDQVWlBRfDmhmrffBD37wNe3zzz+/oErah4PbzHL16quvvqa9f//+gippHw5uM8vVgw8+WLdtM+fgNrNcHXwerV3Pq7WSg9vMcnXwrcoGBgYKqqR9OLjNLFef+tSnXtP2rcuOnIPbzHJXvciUHTkHt5nlqlQq0dXVBVSWVfAXcI6cg9vMcuUVKJvPwW1mufLqgM3n4DazXA0NDR2Y254zZ47XxGkCB7eZ5aqnp4eTTz4ZgJNPPtkrUDaBg9vMclUul9m9ezcAu3btolwuF1xR+hzcZparW265pW7bZs7BbWa52rJlS922zZyD28wsMQ5uM7PEOLjNLFfHHnts3bbNnIPbzHI1OTlZt20z5+A2s1ydcsopdds2cw5uM8vVrl276rZt5nILbkl3SNor6fGqvpMkbZT0dPZ8YtVrN0gak/SUpEuq+s+VtC177WZ5XUizpHiqpPnyHHHfCQwe1Hc9sDkiFgGbszaSzgSWA2dlx9wqqSs75jZgGFiUPQ5+TzObxaaXdD1U22Yut+COiPuB3x7UfRlQyrZLwOVV/WsjYn9EPAOMAUskzQOOj4iHonKjujVVx5hZAs4///y6bZu5Vs9xnxoRuwGy5+mzFH3As1X77cz6+rLtg/trkjQsaVTS6Pj4eFMLN7PD45sDN99sOTlZa9466vTXFBGrIqI/Ivp7e3ubVpyZHb4HH3zwNe0HHnigoEraR6uDe082/UH2vDfr3wksqNpvPrAr659fo9/MEjE1NVW3bTPX6uBeDwxl20PAXVX9yyXNlXQ6lZOQD2fTKfsknZddTbKi6hgzS8CcOXPqtm3m8rwc8LvAQ8AZknZKuhL4MrBM0tPAsqxNRGwH1gFPACPANRExfc3Q1cBqKicsfwncm1fNZtZ8PjnZfN15vXFEXHGIly4+xP43ATfV6B8Fzm5iaWbWQm9605te0547d25BlbQP/5vFzHLlk5PN5+A2s1wtWbKkbttmzsFtZrl68skn67Zt5hzcZparPXv21G3bzDm4zcwS4+A2M0uMg9vMLDEObqNcLrNy5UrK5XLRpZhZAxzcRqlUYtu2baxZs6boUsysAQ7uDlculxkZGSEiGBkZ8ajbLAEO7g5XKpUOrNY2OTnpUbc1ne+A03wO7g63adMmJiYmAJiYmGDjxo0FV2TtxvecbD4Hd4cbGBigu7uy1lh3dzfLli0ruCIzeyMO7g43NDR0YKpkamqKFStWFFyRmb0RB7eZWWIc3B2uVCpRubkQSPLJSbMEOLg73KZNmw6cLJqcnPTJSbMEOLg7nE9OmqXHwd3hhoaGDty8tauryycnzRLg4O5wPT09LF26FIClS5fS09NTbEFm9oYc3Hbg5KSZpcHB3eHK5TL33XcfAFu2bPFaJWYJcHB3OK9VYpYeB3eH81olZulxcHe4888/v27bzGYfB3eHi4iiSzCzGXJwd7gHHnjgNe3777+/oErMrFGFBLekHZK2SXpM0mjWd5KkjZKezp5PrNr/Bkljkp6SdEkRNberU089tW7bzGafIkfcF0bE4ojoz9rXA5sjYhGwOWsj6UxgOXAWMAjcKsm30GiS559/vm7bzGaf2TRVchlQyrZLwOVV/WsjYn9EPAOMAUtaX157eutb31q3bWazT1HBHcD/lvSopOGs79SI2A2QPZ+S9fcBz1YduzPrex1Jw5JGJY2Oj4/nVHp72bNnT922mc0+RQX3ByLifcCHgWskXVBn31rfx655KURErIqI/ojo7+3tbUadbe/gy/8uuKDefwozmw0KCe6I2JU97wW+T2XqY4+keQDZ895s953AgqrD5wO7Wldte3v11Vdf096/f39BlZhZo1oe3JKOk/SW6W3gj4HHgfXAULbbEHBXtr0eWC5prqTTgUXAw62tun0dfPmfLwc0m/26C/jMU4HvZyvSdQPfiYgRSY8A6yRdCfwG+ARARGyXtA54ApgAromIyQLqbksHfwHHX8gxm/1aHtwR8SvgPTX6y8DFhzjmJuCmnEvrSHPmzDlw67LptpnNbv4r7XADAwN122Y2+zi4O9zw8HDdtpnNPg5uM7PEOLg73Je+9KW6bTObfRzcHe7RRx+t2zaz2cfBbWaWGAe3mVliHNxmZolxcJuZJcbBbWaWGAe3mVliHNxmZolxcJuZJcbBbWaWGAe3mVliHNxmZolxcJuZJcbBbWaWGAe3mVliHNxmZolxcJuZJcbBbWaWGAe3mVliHNxmZonpLroAM2udW265hbGxsaLL4Nprr23p5y1cuJDPfvazLf3MPHnEbWaWGI+4zTpIEaPOpUuXvq7vG9/4RsvraCcO7lnE/4ztDLPlv3ORWv07VpS8freTCW5Jg8A3gC5gdUR8ueCSzA7L2NgYT2//KW9/82TRpbTEH54AP3+xu6o9wf5fjxZXUIv85qWu3N47ieCW1AV8E1gG7AQekbQ+Ip5o9mddddVV7N69u9lvm5RWjwbHxsYYGRlp6WcCzJs3j9WrV7f8c5977jkiWv6xBZuickptquhCWiai8t86D0kEN7AEGIuIXwFIWgtcBjQ9uF988UVefvnlZr9tUjrl53/xxRcL++z9k+LX+/IbkR3KP06JqQL+pzGnausXL7b+mog5gqPmtPYH3z8pjsvpvVMJ7j7g2ar2TuD9B+8kaRgYBnj7299+WB/0oQ99qLD5x+eee47f//73Lf/c6qA+7ri8ftUO7ZhjjqGvr6/ln7tw4cKWfyZ05u9Y0drtd0yRwL/ZJH0CuCQirsranwSWRMQhZ/37+/tjdLT959Gaofqs/5YtWwqrw8xeR7U6Uxlx7wQWVLXnA7sKqqXtOKzN0pLKF3AeARZJOl3Sm4DlwPqCazIzK0QSI+6ImJD0GeAHVC4HvCMithdclplZIZIIboCI2ABsKLoOM7OipTJVYmZmGQe3mVliHNxmZolxcJuZJcbBbWaWGAe3mVliHNxmZolxcJuZJcbBbWaWmCRWBzwcksaBXxddR0JOBl4oughra/4dm7kXImLw4M62DW6bGUmjEdFfdB3Wvvw71jyeKjEzS4yD28wsMQ5um7aq6AKs7fl3rEk8x21mlhiPuM3MEuPgNjNLjIPbkDQo6SlJY5KuL7oeay+S7pC0V9LjRdfSLhzcHU5SF/BN4MPAmcAVks4stiprM3cCr/sSiR0+B7ctAcYi4lcR8SqwFris4JqsjUTE/cBvi66jnTi4rQ94tqq9M+szs1nKwW2q0edrRM1mMQe37QQWVLXnA7sKqsXMGuDgtkeARZJOl/QmYDmwvuCazKwOB3eHi4gJ4DPAD4AngXURsb3YqqydSPou8BBwhqSdkq4suqbU+SvvZmaJ8YjbzCwxDm4zs8Q4uM3MEuPgNjNLjIPbzCwxDm7raJJOkPTpFnzO5V68y5rFwW2d7gSg4eBWxeH83VxOZfVFsyPm67ito0maXg3xKeA+4N3AicBRwI0RcZek04B7s9f/BZUQXgH8KZUFul4AHo2Ir0h6J5VlcnuBV4B/D5wE3AP8Q/b4eET8skU/orWh7qILMCvY9cDZEbFYUjdwbET8TtLJwI8kTX/9/wzgzyLi05L6gY8D76XyN/QT4NFsv1XAn0fE05LeD9waERdl73NPRPzPVv5w1p4c3Gb/RMCXJF0ATFFZ3vbU7LVfR8SPsu0PAndFxO8BJN2dPb8Z+CPgf0gHFl2c26LarYM4uM3+yZ9SmeI4NyL+UdIO4OjstZer9qu1FC5Uzhm9GBGLc6vQDJ+cNNsHvCXb/mfA3iy0LwT+4BDHPAj8K0lHZ6PsjwBExO+AZyR9Ag6cyHxPjc8xOyIObutoEVEGfpjdyHYx0C9plMro++eHOOYRKkvf/gz4HjBK5aQj2XFXSvoZsJ1/ug3cWuA/SPppdgLT7LD5qhKzwyDpzRHxkqRjgfuB4Yj4SdF1WWfwHLfZ4VmVfaHmaKDk0LZW8ojbzCwxnuM2M0uMg9vMLDEObjOzxDi4zcwS4+A2M0vM/wclvsiNTFf1LQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data = df, x = \"target\", y = \"personal\", kind = \"box\")\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "436fb837",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data = df, x = \"target\", y = \"medios\", kind = \"box\")\n",
    "plt.show() "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e75786b",
   "metadata": {},
   "source": [
    "Porcentaje de incendios con impacto significativo según la comunidad y según la causa."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "49e2fe9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "freq_comunidad = pd.DataFrame(pd.crosstab(df[\"idcomunidad\"], df[\"target\"], normalize='index'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "22d36bfb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.bar(freq_comunidad.index, freq_comunidad.values[:, 1])\n",
    "plt.xlabel(\"Identificador de comunidad\")\n",
    "plt.ylabel(\"% Incendios con impacto significativo\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "fd10889a",
   "metadata": {},
   "outputs": [],
   "source": [
    "freq_causa = pd.DataFrame(pd.crosstab(df[\"causa\"], df[\"target\"], normalize='index'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "6a6be0bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.bar(freq_causa.index, freq_causa.values[:, 1])\n",
    "plt.xlabel(\"Identificador de causa\")\n",
    "plt.ylabel(\"% Incendios con impacto significativo\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bda4adde",
   "metadata": {},
   "source": [
    "Base de datos no balanceada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "17aa2aaf",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    61365\n",
       "1    21273\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.target.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "279e5a7a",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.25742394540042113"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(df.target == 1)/len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "bb12532a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7425760545995789"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sum(df.target == 0)/len(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2bdcb26",
   "metadata": {},
   "source": [
    "División entrenamiento y test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "77a7f55d",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(df, df[\"target\"], test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "42685cd5",
   "metadata": {},
   "source": [
    "DownSampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "d2024e8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    49048\n",
       "1    17062\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.target.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "fd5eeb14",
   "metadata": {},
   "outputs": [],
   "source": [
    "majority = x_train[x_train.target==0]\n",
    "minority = x_train[x_train.target==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "0c4a9111",
   "metadata": {},
   "outputs": [],
   "source": [
    "majority_downsampled = resample(majority, \n",
    "                                replace = False,       # sample without replacement\n",
    "                                n_samples = 16987,     # to match minority class\n",
    "                                random_state = 0)    # reproducible results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "2dc0a589",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_downsampled = pd.concat([majority_downsampled, minority])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e5bd7769",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    17062\n",
       "0    16987\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train_downsampled.target.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "642472c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_downsampled = x_train_downsampled.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "8a4f62da",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_downsampled.drop(\"target\", axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b5926f0c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>superficie</th>\n",
       "      <th>idcomunidad</th>\n",
       "      <th>idprovincia</th>\n",
       "      <th>idmunicipio</th>\n",
       "      <th>causa</th>\n",
       "      <th>time_ctrl</th>\n",
       "      <th>time_ext</th>\n",
       "      <th>personal</th>\n",
       "      <th>medios</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>40335</th>\n",
       "      <td>3.00</td>\n",
       "      <td>3</td>\n",
       "      <td>32</td>\n",
       "      <td>16</td>\n",
       "      <td>4</td>\n",
       "      <td>225</td>\n",
       "      <td>240</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40214</th>\n",
       "      <td>3.00</td>\n",
       "      <td>17</td>\n",
       "      <td>49</td>\n",
       "      <td>145</td>\n",
       "      <td>4</td>\n",
       "      <td>375</td>\n",
       "      <td>390</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55126</th>\n",
       "      <td>5.00</td>\n",
       "      <td>17</td>\n",
       "      <td>49</td>\n",
       "      <td>65</td>\n",
       "      <td>4</td>\n",
       "      <td>100</td>\n",
       "      <td>470</td>\n",
       "      <td>32</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>61282</th>\n",
       "      <td>7.00</td>\n",
       "      <td>5</td>\n",
       "      <td>33</td>\n",
       "      <td>52</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>278</td>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64002</th>\n",
       "      <td>8.16</td>\n",
       "      <td>4</td>\n",
       "      <td>18</td>\n",
       "      <td>47</td>\n",
       "      <td>2</td>\n",
       "      <td>485</td>\n",
       "      <td>1895</td>\n",
       "      <td>113</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68755</th>\n",
       "      <td>12.00</td>\n",
       "      <td>14</td>\n",
       "      <td>10</td>\n",
       "      <td>147</td>\n",
       "      <td>4</td>\n",
       "      <td>290</td>\n",
       "      <td>1590</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>82457</th>\n",
       "      <td>971.66</td>\n",
       "      <td>2</td>\n",
       "      <td>17</td>\n",
       "      <td>210</td>\n",
       "      <td>4</td>\n",
       "      <td>1184</td>\n",
       "      <td>5481</td>\n",
       "      <td>584</td>\n",
       "      <td>127</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>80186</th>\n",
       "      <td>90.00</td>\n",
       "      <td>5</td>\n",
       "      <td>33</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>1887</td>\n",
       "      <td>3327</td>\n",
       "      <td>34</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48600</th>\n",
       "      <td>4.00</td>\n",
       "      <td>3</td>\n",
       "      <td>32</td>\n",
       "      <td>63</td>\n",
       "      <td>4</td>\n",
       "      <td>162</td>\n",
       "      <td>177</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>80471</th>\n",
       "      <td>100.00</td>\n",
       "      <td>11</td>\n",
       "      <td>13</td>\n",
       "      <td>87</td>\n",
       "      <td>2</td>\n",
       "      <td>427</td>\n",
       "      <td>1531</td>\n",
       "      <td>73</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>34049 rows × 9 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       superficie idcomunidad idprovincia idmunicipio causa  time_ctrl  \\\n",
       "40335        3.00           3          32          16     4        225   \n",
       "40214        3.00          17          49         145     4        375   \n",
       "55126        5.00          17          49          65     4        100   \n",
       "61282        7.00           5          33          52     4          0   \n",
       "64002        8.16           4          18          47     2        485   \n",
       "...           ...         ...         ...         ...   ...        ...   \n",
       "68755       12.00          14          10         147     4        290   \n",
       "82457      971.66           2          17         210     4       1184   \n",
       "80186       90.00           5          33          22     4       1887   \n",
       "48600        4.00           3          32          63     4        162   \n",
       "80471      100.00          11          13          87     2        427   \n",
       "\n",
       "       time_ext  personal  medios  \n",
       "40335       240         8       1  \n",
       "40214       390        12       0  \n",
       "55126       470        32       4  \n",
       "61282       278         6       1  \n",
       "64002      1895       113      18  \n",
       "...         ...       ...     ...  \n",
       "68755      1590        22       4  \n",
       "82457      5481       584     127  \n",
       "80186      3327        34       4  \n",
       "48600       177        11       1  \n",
       "80471      1531        73      17  \n",
       "\n",
       "[34049 rows x 9 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train_downsampled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "e73a3cae",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/usuario/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py:4308: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  return super().drop(\n"
     ]
    }
   ],
   "source": [
    "x_train.drop(\"target\", axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "c68de2e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_test.drop(\"target\", axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8bde0a1",
   "metadata": {},
   "source": [
    "Definición métricas de los modelos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "afef6e1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def metricas_modelos(y_true, y_pred):\n",
    "    \n",
    "    conf_matrix = confusion_matrix(y_true, y_pred)\n",
    "\n",
    "    print('La matriz de confusión es ')\n",
    "    print(conf_matrix)\n",
    "\n",
    "    print('Precisión:', accuracy_score(y_true, y_pred))\n",
    "    print('Exactitud:', precision_score(y_true, y_pred))\n",
    "    print('Exhaustividad:', recall_score(y_true, y_pred))\n",
    "    print('F1:', f1_score(y_true, y_pred))\n",
    "\n",
    "    false_positive_rate, recall, thresholds = roc_curve(y_true, y_pred)\n",
    "    roc_auc = auc(false_positive_rate, recall)\n",
    "\n",
    "    print('AUC:', auc(false_positive_rate, recall))\n",
    "\n",
    "    plot(false_positive_rate, recall, 'b')\n",
    "    plot([0, 1], [0, 1], 'r--')\n",
    "    title('AUC = %0.2f' % roc_auc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "171e5052",
   "metadata": {},
   "source": [
    "KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "9abb30e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "neighbors = range(2,16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "8423c92b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=10, estimator=KNeighborsClassifier(),\n",
       "             param_grid={'n_neighbors': range(2, 16)})"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "modelo_knn = KNeighborsClassifier()\n",
    "\n",
    "grid = GridSearchCV(estimator = modelo_knn,\n",
    "                    param_grid = dict(n_neighbors = neighbors),\n",
    "                    cv = 10)\n",
    "\n",
    "grid.fit(x_train_downsampled, y_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "dfd82bda",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "El mejor parametro es {'n_neighbors': 15}\n"
     ]
    }
   ],
   "source": [
    "print('El mejor parametro es', grid.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "1328324d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=15)"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "modelo_knn = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'])\n",
    "modelo_knn.fit(x_train_downsampled, y_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "739c1f5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_knn_train_downsampled = modelo_knn.predict(x_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "0af11d44",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_knn_train = modelo_knn.predict(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "e2b46560",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_knn_test = modelo_knn.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "620109f5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "La matriz de confusión es \n",
      "[[13071  3916]\n",
      " [ 6166 10896]]\n",
      "Precisión: 0.7038973244441834\n",
      "Exactitud: 0.7356197677558736\n",
      "Exhaustividad: 0.6386121205016997\n",
      "F1: 0.6836920373972516\n",
      "AUC: 0.7040414461341725\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metricas_modelos(y_train_downsampled, pred_knn_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ebbec230",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "La matriz de confusión es \n",
      "[[8855 3462]\n",
      " [1727 2484]]\n",
      "Precisión: 0.6860479186834463\n",
      "Exactitud: 0.4177598385469223\n",
      "Exhaustividad: 0.5898836380907148\n",
      "F1: 0.4891208033868268\n",
      "AUC: 0.6544043505059404\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metricas_modelos(y_test, pred_knn_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4004f6c6",
   "metadata": {},
   "source": [
    "Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "55e52e9c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(max_depth=10, random_state=0)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = RandomForestClassifier(max_depth = 10,\n",
    "                            n_estimators = 100,\n",
    "                            random_state = 0)\n",
    "\n",
    "clf.fit(x_train_downsampled, y_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "582989ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_clf_train_downsampled = clf.predict(x_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "7622967c",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_clf_train = clf.predict(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "6bdc0433",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_clf_test = clf.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "1ead360c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "La matriz de confusión es \n",
      "[[13352  3635]\n",
      " [ 4358 12704]]\n",
      "Precisión: 0.7652500807659549\n",
      "Exactitud: 0.7775261643919457\n",
      "Exhaustividad: 0.7445785957097644\n",
      "F1: 0.7606957875512708\n",
      "AUC: 0.7652957145264546\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metricas_modelos(y_train_downsampled, pred_clf_train_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "fedbd761",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "La matriz de confusión es \n",
      "[[8946 3371]\n",
      " [1294 2917]]\n",
      "Precisión: 0.7177516940948693\n",
      "Exactitud: 0.4638994910941476\n",
      "Exhaustividad: 0.6927095701733555\n",
      "F1: 0.5556719687589294\n",
      "AUC: 0.709511397898239\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEICAYAAABPgw/pAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAArFUlEQVR4nO3deZzW8/rH8delpFKSZGtRKMkhaZRICCXOUdYfIhzJFomD7GsdWY4sFVlOx7F0LCFEkaVOiSYqJSWhkmjRvs7M5/fHNXNmjKnuqfu+v/fyfj4e98Pc3/s79319m3Ou+cz1/Xyuj4UQEBGR9Ldd1AGIiEh8KKGLiGQIJXQRkQyhhC4ikiGU0EVEMoQSuohIhlBCFxHJEErokpLM7GMz+83MdijjeLdSx44xs/klnpuZXW1m08xstZnNN7NXzOygOMe4i5m9XvgZP5rZuZs59wkzW1Xisd7MVpZ4vYeZ5RYeHxLPOCV7KKFLyjGzBsBRQABO2Yq3eAToCVwN7AI0Bt4ATo5PhP8zANgA7A50AQaZ2YFlnRhCuCyEUK3oAbwEvFLilAXAvcCzcY5RskjFqAMQKUNXYALwGXABv098m2VmjYArgdYhhM9LvPRCPAM0sx2B04E/hRBWAf81s+HA+UDvGL/3z0XHQgjDCl/LAerGM1bJHhqhSyrqiifgF4AOZrZ7Ob73OGB+qWS+WWY20MyWbeIxdRPf1hjIDyHMKnFsClDmCL2U04FFwJhYYxSJhRK6pBQzawPsDbwcQpgEfAdssjZdhlrAz+X5zBDCFSGEnTfxOHgT31YNWF7q2HKgegwfeQHwXFAjJYkzJXRJNRcAo0IIiwufv1h4rEgesH2p79ke2Fj49RJgz4RG6FYBO5U6thOwsoxz/8fM6gFHA88lKC7JYkrokjLMrApwFnC0mS00s4VAL6CZmTUrPG0u0KDUtzYEfiz8ejRQt7AWHevnlp6BUvIxfRPfNguoWFizL9IM2NT5RboC40MIc2KNTyRWSuiSSjoD+UBT4JDCxwHAWDwRAvwHuMjMWhZOT2yMJ/2hACGEb4GBwEuF0xkrmVllMzvbzMq8WVl6Bkqpx6ZmrawGhgF3m9mOZnYk0An49xausSswpPRBM6toZpWBCkCFwpg1aUHKJ4Sghx4p8QDeAx4q4/hZwEKgYuHzv+Ij4RXAbHxWyXYlzjd82uJ0YA3wE/6L4MA4x7sLPh1yNf6Xw7klXquPl2XqlzjWuvDc6mW81534NM2Sjzuj/pnokV4PC0H3ZUREMoFKLiIiGUIJXUQkQyihi4hkCCV0EZEMEdm0qF133TU0aNAgqo8XEUlLkyZNWhxCqF3Wa5El9AYNGpCbmxvVx4uIpCUz+3FTr6nkIiKSIZTQRUQyhBK6iEiGUEIXEckQSugiIhliiwndzJ41s1/NbNomXjcze9TMZpvZVDM7NP5hiojIlsQyQh8CnLiZ1zsCjQof3YFB2x6WiIiU1xYTeghhDLB0M6d0onA7rRDCBGBnM0vGjjEiImmjoAAmT9zIU9fP4sMPE/MZ8aih1wHmlXg+v/DYH5hZdzPLNbPcRYsWxeGjRURS18KF8O9/w/nnQ/vaXxJatuSkB4/l43dWJ+Tz4rFS1Mo4VmaT9RDCYGAwQE5Ojhqxi0hGWbcOxo2DkSNh1CiYMgV2YB39qtxFj3UPsGGnXVn70EDu7rZjQj4/Hgl9PlCvxPO6wII4vK+ISEoLAWbM8OQ9ciR88gmsXQvbbw9HHgl9+8IVb3Wmxqcj4aKLqPLQQ1SpWTNh8cQjoQ8HepjZUKAVsDyE8HMc3ldEJOUsWQIffOBJfNQomD/fjzduDN26Qfv2cEyLlVSruT1Urgyte8PG6+CEExIe2xYTupm9BBwD7Gpm84E7gO0BQghPACOAk/C9HdcAFyUqWBGRZNu4ET79tDiB5+b6yHznneG44+D22z1X/6957MiRcHh3OO886NMHjjkmabFuMaGHEM7ZwusBuDJuEYmIRCgE+O674jr4hx/CqlVQoQK0agV33AEdOkBODlQsmUGXLoVrr4V//QuaNIGTT0567JG1zxURSRXLl3viLkri33/vxxs0gC5dvIzSrp2Pyss0erSfuGQJ3HIL3Hqrl1uSTAldRLJOfj5MnFh8M/Ozz/xYtWqeuK+7zpP4fvuBlTWPr7TddoOGDeG99+CQQxId/iYpoYtIVvjxx+I6+AcfwLJlnqxbtIDevT2BH344VKoUw5uF4KWVL76ARx+Fgw6C8eNjzP6Jo4QuIhlp1SqfRlhURpk504/XqQOnnup18OOOg113Lecbf/89XHopvP8+HHWUz1OsUiXyZA5K6CKSIQoKYPLk4jLKuHE+Q6VKFTj6aM/BHTrAAQdsZe7Nz4cBA+Cmm2C77WDgQH/T7VKnaa0SuoikrQULfKA8apT/t6ijSLNmcM01XkZp0yZO9ycXL/Y5ikcfDU88AfXrx+FN40sJXUTSxtq1MHZs8Sh8WmFT791289F3+/Zw/PGwZ7zaA27cCC+8AF27wu67e828YcOUKK+URQldRFJWCDB9enEdfMwY75dSqZKPvPv18yR+8MEJqHxMmgR//StMneq/ITp0gH32ifOHxJcSuoiklEWLfBZKURL/ubCRyAEHFNfB27aFHRPT38r/DLjrLnjwQR/6v/66f2gaUEIXkUht2OAz/orKKF984cdr1vQl9e3b+3+TVrLu3NmD6dYNHnhgM6uJUo/5yv3ky8nJCbm5uZF8tohEJwSYNat4TvhHH8Hq1b6MvnVrT+Dt2/v88AoVkhTUihVex6lc2ec65uX5nMYUZGaTQgg5Zb2mEbqIJNxvv/nq+KIk/uOPfnzffeGCCzyBH3ss7LRTBMGNGAGXXebNtPr29VksaUoJXUTiLi8PPv+8uA7++ec+T7x6dR/49u7tZZR9940wyMWLoVcveP55aNoUTjklwmDiQwldROLi+++LR+CjR3vDq+22g8MO835V7dt7t8Ltt486UnzSepcu/qfD7bfDzTfDDjtEHdU2U0IXka2ycqXXv4uS+Lff+vF69eDMMz2BH3cc7LJLtHGWac89fUeKQYO8D0uGUEIXkZgUFPgMlKIyyvjxXlqpWtX3cOjRw5P4/vun4LqbEOCZZ+DLL335/p/+5CuUUi7QbaOELiKbNH++VydGjvS54UuW+PHmzb3FbIcOcMQRKV6tmDMHLrnEG54fc0xKNdOKNyV0EfmfNWt8NWbRnPCvv/bje+zhG/AUzQnfbbdo44xJfr63tr3lFp8T+eSTPrc8hZppxZsSukgWC8FXthfVwceOhfXrfcTdti1cdJEn8YMOSsMB7eLFvuLzuOO8Vl63btQRJZwSukiW+eWX33coXLjQjx94IFx5pSfwo47y2nja2bDBpyFeeKE305o8GfbeOw1/G20dJXSRDLd+vfcGL7qZOXmyH69Vy8snHTr4f+vUiTTMbTdxojfTmjbNR+Pt2/umoFlECV0kw4QA33xTXAf/5BOvjVesCEceCX36eBJv3jxDyslr1vhc8ocf9umIw4d7Ms9CSugiGWDpUp+FUlQLnzfPjzdu7IPW9u19gkf16pGGmRidOvnFd+8O998PNWpEHVFk1JxLJA1t3AgTJhQn8IkTfWReo4Zv8FA0G6Vhw6gjTZDly/3ObeXKPi0nP9+bwWQBNecSyQDffVdcRvnwQ1+pud12vpz+jjs8iR92mJdWMtrbb3szrfPPh7//3afjCKCELpKyli8vXlo/cqSvjwGftHHOOV4Hb9curdp1b5tFi6BnT3jpJZ9HedppUUeUcpTQRVJEfj7k5haXUT791I/tuKMn7l69PInvt1/WzMIrNmqUN9Navtznlvfu7f3L5XeU0EUiNG9e8XTCDz7w5n9mvrnDjTd6GaV1a+Uu6tTxPegGDfIJ81ImJXSRJFq92qcRFiXxb77x43vt5TufFXUorF070jCjV1AATz/tzbSKkviYMVFHlfKU0EUSqKAApkwproOPG+eLGStX9o1xLrnEyyhNm2ZhGWVTZs/2f5iPP/aZK0XNtGSLlNBF4uznn3+/tP7XX/34wQfD1VcXL62vXDnaOFNOfj707w+33ea7YDz1FFx8sX7TlUNMCd3MTgQeASoAT4cQ7iv1eg3geaB+4Xs+GEL4Z5xjFUlJ69Z5U6uim5lTp/rx2rWLNzw+4QRfxCibsXgx3Huv/2MNHJgBvQiSb4sJ3cwqAAOAE4D5wEQzGx5C+LrEaVcCX4cQ/mJmtYGZZvZCCGFDQqIWiVAI3la2qA7+ySee1CtVgjZt4L77PIk3a5YhS+sTaf16eO45H4kXNdOqX1+j8q0Uywi9JTA7hDAHwMyGAp2Akgk9ANXNzIBqwFIgL86xikRm8WKfhVKUxBcs8ONNmviK8w4dvCa+447RxplWPvvME/n06T65vn17/69stVgSeh1gXonn84FWpc55HBgOLACqA/8XQigo/UZm1h3oDlC/fv2tiVckKTZs8HngRWWUSZN8ZF6zZvHS+vbtfTAp5bR6tdfJ+/f3sso772RtM614iyWhl/W3T+kGMB2AyUA7YF/gfTMbG0JY8btvCmEwMBi8l0u5oxVJkBB8ckXRCPyjj2DVKqhQweeB33WX55ycHD8m26BzZ/9z5/LLvT61005RR5QxYkno84F6JZ7XxUfiJV0E3Be809dsM/seaAJ8HpcoRRJg2TLviVKUxH/4wY/vs4+3CWnf3mfNZXHzvvhZtsybaVWp4q1ub7tNPVgSIJaEPhFoZGYNgZ+As4FzS50zFzgOGGtmuwP7A3PiGajItsrL866ERXPCP/vM54lXr+5L62+4wZP4vvtGHWmGGT7cR+Pnn+8j8qOOijqijLXFhB5CyDOzHsBIfNrisyGE6WZ2WeHrTwD3AEPM7Cu8RHNjCGFxAuMWickPPxTXwUeP9oGimXclvPlmv5nZqpVPe5Y4+/VXn3j/n//4JPwzzog6oowX0zz0EMIIYESpY0+U+HoBoLsaErlVq4o7FI4aBbNm+fG6deH004uX1teqFW2cGe+997yZ1qpVcM893phGvzUTTitFJa0VFHi7j6I6+PjxvvlD1aq+Q88VV3gSb9JEU5uTql49b3E7cKD3NZCkUEKXtPPTT76kfuRInyyxuLC4d8ghcO21nsCPPNLvwUmSFBTAk0/6wqAnn/RmWh9/HHVUWUcJXVLe2rXeaK/oZub06X58992hY0evgx9/vD+XCMyaBd26ef+DE07wZbNqVBMJJXRJOSHAtGnFZZQxY3yF+A47+ASJCy7wJH7QQSqjRCovDx56yPe/q1IF/vlP/+HohxIZJXRJCb/++vul9QsX+vGmTYvr4G3bem1cUsSSJdCvH5x0EgwYoO5jKUAJXSKxfr3fwCwqo3z5pR+vVcv/ai/qUFi3brRxSinr18OQId6vfPfdvdl7vXpb/DZJDiV0SYoQYObM4umEH3/sLT0qVoQjjoA+fTyJN2+upfUp69NPvZnWjBm++ur445XMU4wSuiTM0qW+mKcoic+d68cbNYILLyxeWl+9eqRhypasWgW33gqPPuoJ/L33PJlLylFCl7jJy/Pl9EV18IkTfTZbjRq+mOfmmz2JN2wYdaRSLp07+2/mHj2gb1/9Bk5h5v20ki8nJyfk5uZG8tkSP3PmFNfBP/wQVqzwTR1atvSZKO3b+9cVNXRIL7/95lMPq1SB//7Xj7VpE21MAoCZTQoh5JT1mv5vJuWyYkXx0vqRI+G77/z43nvD2Wd7Am/XzvuGS5oaNgyuvBK6dvVZLErkaUMJXTYrPx+++KK4jPLpp15a2XFHr3/37Okj8UaNNP047S1c6GWV117zZbdnnx11RFJOSujyB/Pm/X5p/dKlfrxFC7j+eh+FH3GE76EpGeLdd72Z1po1Xif/29/UTCsNKaELq1f/fmn9jBl+fK+94JRTPIEff7zvYi8Zau+9fc7ogAHeyUzSkhJ6FtuwwQdlw4f715Ur+2rMbt08iR94oMooGaugwDshTpkCTz3lS3JHj446KtlGSuhZ7KGH4NVXvWx6yil+76tKlaijkoSbOdMXCI0b5zdA1EwrYyihZ6kffvB9B04/HR57LOpoJCk2boQHH/Qdr6tW9SX8Xbvqz7AMooSepXr29PniDz8cdSSSNL/9Bg88AH/5i/8W32OPqCOSONsu6gAk+d56y+vmd9yhVhwZb906r5UXFMBuu8HUqfDKK0rmGUoJPcusWeP79jZtCtdcE3U0klD//S80a+aLhD780I+pfWVGU0LPMn37ev180CBNM85YK1f6ne6jjvLpS6NGqZlWllANPYvMnAn33+/3wdq2jToaSZjOnb0/Q8+ecO+9UK1a1BFJkiihZ4kQfNBWtaondckwS5f61MOqVX36khm0bh11VJJkKrlkiZdf9mX8fftqM+WM8+qrcMABcOed/vyII5TMs5QSehZYsQJ69fJeLJdeGnU0Ejc//wynnQZnnunTlbp0iToiiZhKLlngjju8kd6bb2p7t4zxzjtw3nk+LbFfP7j2WjWdFyX0TDdliq8hufRSOOywqKORuNlnH/+BPv44NG4cdTSSIlRyyWAFBXDFFbDLLl47lzSWnw+PPOI9WMBr5qNGKZnL7yihZ7AhQ2D8eF/trR2E0tjXX/uc8muu8drZunVRRyQpSgk9Qy1ZAjfc4Hmga9eoo5GtsmGDzyNv3hxmzYLnn4e331ZnRNmkmBK6mZ1oZjPNbLaZ9d7EOceY2WQzm25mn8Q3TCmvm2+GZct8vwI100tTy5Z597RTT/VRepcu+mHKZm3xpqiZVQAGACcA84GJZjY8hPB1iXN2BgYCJ4YQ5prZbgmKV2IwYYLvWXDttXDQQVFHI+Wydi0884zf/NhtN/jqK986SiQGsYzQWwKzQwhzQggbgKFAp1LnnAsMCyHMBQgh/BrfMCVW+fmeC/bay6crShoZM8abaV11lS/dByVzKZdYEnodYF6J5/MLj5XUGKhpZh+b2SQzK7Nqa2bdzSzXzHIXLVq0dRHLZg0aBF9+Cf37Q/XqUUcjMVmxwn8LH3005OX5kt7jjos6KklDscxDL6toF8p4nxbAcUAV4FMzmxBCmPW7bwphMDAYICcnp/R7yDZauBBuucX3Az399KijkZh17gwff+zLee+5B3bcMeqIJE3FktDnAyW3QagLLCjjnMUhhNXAajMbAzQDZiFJ87e/+Yy2xx/XvbOUt3ixN9KqWhX69PEf2OGHRx2VpLlYSi4TgUZm1tDMKgFnA8NLnfMmcJSZVTSzqkArYEZ8Q5XN+egjeOEF6N0bGjWKOhrZpBBg6FBfGFR0k6N1ayVziYstjtBDCHlm1gMYCVQAng0hTDezywpffyKEMMPM3gOmAgXA0yGEaYkMXIpt2OCb0jRs6AldUtRPP3mtfPhwX7avBQISZzH1cgkhjABGlDr2RKnnDwAPxC80idXDD8OMGd6vqUqVqKORMr39ts8j37gRHnzQV32qU5rEmZpzpbkff4S77/a1JyedFHU0skn77ed9yh97zL8WSQAt/U9zRRs99+8fZRTyB/n5/qfThRf68yZN4N13lcwloZTQ09g778Abb8Dtt0P9+lFHI/8zfToceaQv1V28WM20JGmU0NPU2rW+oPCAA3z6sqSADRu8/tW8OXz3Hbz4Irz1lpppSdKohp6m/v53+P57n65YqVLU0QjgzbQefdS3hOvfH2rXjjoiyTIaoaehb7/1XcfOOw+OOSbqaLLcmjW+8UR+fnEzrRdeUDKXSCihp5kQoEcP/yv+AU0SjdZHH3k7y2uu8aX7AHvuGWVEkuWU0NPMq6/6zmN9+sAee0QdTZZavtw3aW3Xzpfsf/SRmmlJSlANPY2sXOmDwebN4fLLo44mi3Xu7K1ur78e7rzT+7GIpAAl9DRy553w888wbJgWGSbdokXeBbFqVb8jXaGCL98XSSEquaSJr77ye2+XXAKtWkUdTRYJwacflmymdfjhSuaSkpTQ00BBgZdYataEvn2jjiaLzJ8Pp5ziPVj226941adIilLJJQ089xyMGwfPPgu1akUdTZYYPtznhRYt4b/qKtW5JOUpoae4pUvhhhu8r9MFF0QdTRZp3BjatPHdQvbZJ+poRGKikkuKu+UWT+qDBsF2+mklTl6et7Ut6lHepAmMGKFkLmlFKSKFff45PPkkXH01HHxw1NFksKlTfdeg66/3DZvVTEvSlBJ6isrP9xuhe+7p0xUlAdav95krLVrA3Lnw8svw+utqpiVpSzX0FPXkk/DFF7795E47RR1NhlqxAgYOhHPO8RufuuMsaU4j9BT0yy9w881w/PFw1llRR5NhVq/25J2f7w20pk3zaURK5pIBlNBT0PXXe7/zAQO8VYjEyejR3kzr2mvhk0/82O67RxuTSBwpoaeYTz6Bf//bpyo2bhx1NBli2TLo1s3/5KlY0f+R27WLOiqRuFMNPYVs3AhXXAENGsBNN0UdTQY59VQYOxZuvNFvglapEnVEIgmhhJ5C+veHr7/2XcvUwG8b/fILVKvmDbXuu89H5i1aRB2VSEKp5JIi5s3z6YmdOsGf/xx1NGksBK9ZNW1a3EyrVSslc8kKSugp4pprPBc98kjUkaSxuXPh5JN9tef++8PFF0cdkUhSqeSSAt5913uc9+0Le+8ddTRp6s03vZlWCL5R8xVXqJmWZB0l9IitXet7hDZpAtddF3U0aSgEn9vZpInvmP3YY35XWSQLKaFHrF8/mDPHp0hXqhR1NGkkLw8eesh3/nj+eS+xvPVW1FGJREo19AjNnu0TMM49V9Oiy2XKFL/R2bs3rFmjZloihZTQIxKCl1p22MG7tkoM1q2DW2+FnBz46Sd49VW/+aBmWiKASi6RGTYMRo70WS177hl1NGli5UrvWtalC/zjH7DLLlFHJJJSYhqhm9mJZjbTzGabWe/NnHeYmeWb2RnxCzHzrFwJPXvCIYf4ZAzZjFWr/E+YomZaX38NQ4YomYuUYYsJ3cwqAAOAjkBT4Bwza7qJ8/oBI+MdZKa5+26vGAwc6AsYZRNGjYI//ckb24wZ48dq1442JpEUFssIvSUwO4QwJ4SwARgKdCrjvKuA14Bf4xhfxpk2zZf4d+vmm+RIGZYuhYsugg4dvD4+diwce2zUUYmkvFgSeh1gXonn8wuP/Y+Z1QFOBZ7Y3BuZWXczyzWz3EWLFpU31rQXgpdYatTw2S2yCaee6sv3b74ZJk+GI4+MOiKRtBDLH/xldeQOpZ73B24MIeTbZhp4hxAGA4MBcnJySr9Hxvv3v32w+fTT2k/hDxYuhOrVvZnWAw/4pPxDDok6KpG0EssIfT5Qr8TzusCCUufkAEPN7AfgDGCgmXWOR4CZ4rfffOOK1q29miCFQvCbnE2bwu23+7GWLZXMRbZCLCP0iUAjM2sI/AScDZxb8oQQQsOir81sCPB2COGN+IWZ/m69FRYv9qmK22n2v/vhB7j0Ur/52aYNdO8edUQiaW2LCT2EkGdmPfDZKxWAZ0MI083sssLXN1s3F8jNhUGD4OqrNfD8n9dfh/PP9z4sjz8Ol1+u33Qi28hCiKaUnZOTE3JzcyP57GTKz4fDD/dpit98AzvtFHVEEStqpjVrlk9HfOQRtZgUKQczmxRCyCnrNc2CTrCnnvIR+osvZnky37jRb3ZOm+b/GI0bwxtvRB2VSEbR37gJ9Ouvvjdou3Zw9tlRRxOhL77wG5233OJ/sqxfH3VEIhlJCT2BbrgBVq+GAQO8ypB11q7132gtW/q0xNdfh//8xzuSiUjcKaEnyNix8K9/+VTFJk2ijiYiq1fDM8/ABRd4D5bOnaOOSCSjKaEnwMaNviJ07729ypBVVq6E++/30squu3oif+YZqFkz6shEMp5uiibAo4/6vb8334SqVaOOJonee8/nlc+b52WWY47xpC4iSaERepzNnw933AF/+QucckrU0STJkiVeVunY0ZfujxvnyVxEkkoj9Djr1curDY88EnUkSXTaaTB+PNx2m9eYdNNTJBJK6HE0cqTvinbvvdCw4ZbPT2s//+zNtKpV8w0oKlWCZs2ijkokq6nkEifr1vkeoY0bw9/+FnU0CRQCPPssHHBAcTOtww5TMhdJARqhx8n998Ps2fD++xlccZgzx296fvABtG0Ll10WdUQiUoISehx89x307Qv/939w/PFRR5Mgw4Z5M60KFbzTWPfuaqYlkmKU0LdRCN5FsVIl34g+4xQ10zroIDjxRN8/r169LX6biCSfEvo2euMNGDECHn4Y9tor6mjiaMMGryNNn+7NtBo1gtdeizoqEdkM/c28DVatgp494eCD/YZoxsjN9Rudt93mzzdsiDYeEYmJEvo2uOceXxQ5aBBUzIS/ddau9Y5irVr59kpvvgkvvZTBd3lFMosS+lb6+muvmf/1r3DEEVFHEyerV/v+nhdf7KWWrFnqKpIZlNC3QgjefGunnaBfv6ij2UYrVsB99xU305oxAwYPhp13jjoyESknJfSt8MIL8MknngfTuvfUO+/AgQf6cv2xY/1YrVrRxiQiW00JvZyWLfOVoK1aeWUiLS1aBF26wJ//DDVqeB8WNdMSSXuZcCsvqW67zfPhiBFpvK7m9NNhwgS4807fUahSpagjEpE4UEIvh0mTYOBAuPJKOPTQqKMpp59+8tF4tWo+aX6HHeBPf4o6KhGJo3QdYyZdQYHfCK1d26crpo0Q4KmnoGnT4mZaLVoomYtkII3QY/T00/D55/D88z7QTQvffQeXXAIffQTHHut/WohIxtIIPQaLFkHv3n7f8Nxzo44mRq++6v1XJk3yaYijR8O++0YdlYgkkEboMbjxRt/7eOBA71OV0oqaaTVrBief7PXyunWjjkpEkkAj9C0YNw7++U+47jrf0yFlbdgAd90FZ5/tSb1RI3jlFSVzkSyihL4ZeXlw+eXeLbaoT1VK+vxzv9F5553eVEbNtESykhL6Zjz2GHz1FTz6qG9mn3LWrPFVTq1bw2+/wVtv+TJWNdMSyUpK6Jvw008+y+/kk6FTp6ij2YS1a33aTffu3i3sz3+OOiIRiVBMCd3MTjSzmWY228x6l/F6FzObWvgYb2Zpv2Pwtdd6yeXRR1PsRujy5dCnjwdXq5Y30xo0yDuFiUhW22JCN7MKwACgI9AUOMfMmpY67Xvg6BDCwcA9wOB4B5pM778PL78MN98M++wTdTQlvPVW8QKh//7Xj9WsGW1MIpIyYhmhtwRmhxDmhBA2AEOB3xUhQgjjQwi/FT6dAKTt1Ir16339TaNGcP31UUdTaNEiOOcc709eqxZ89pmaaYnIH8QyD70OMK/E8/lAq82cfzHwblkvmFl3oDtA/fr1YwwxuR54AL79FkaOhMqVo46mUFEzrbvv9knxaqYlImWIJaGXVUEOZZ5odiye0NuU9XoIYTCF5ZicnJwy3yNK33/v5ekzz4T27SMOZv5832SiWjXo399nrhx4YMRBiUgqi6XkMh+oV+J5XWBB6ZPM7GDgaaBTCGFJfMJLnhDgqqt8GvfDD0cYSEEBPPmk18qLJr8feqiSuYhsUSwJfSLQyMwamlkl4GxgeMkTzKw+MAw4P4QwK/5hJt7w4b6Bz113QZ06EQXx7bfQrh1cdhm0bOm/YUREYrTFkksIIc/MegAjgQrAsyGE6WZ2WeHrTwC3A7WAgeZz/PJCCDmJCzu+Vq+Gq6/2XlaR5dBXXoGuXb208swzcNFFKTZfUkRSXUzNuUIII4ARpY49UeLrbkC3+IaWPH36wNy5vq3m9tsn+cOLmmk1b+4rmP7xD9hrryQHISKZIOtXis6YAQ8+CBdeCG3KvJWbIOvX+3zys87ypL7ffjB0qJK5iGy1rE7oIfic82rV4P77k/jBEyb4jc577oEqVdRMS0TiIqsT+ksv+WY+ffv61nIJt3o19OoFRxzhDdZHjIDnnlMzLRGJi6xN6MuXe4/zww7zXdqSYt06L6tccQVMnw4dOybpg0UkG2TtjkW33w6//AJvvw0VKiTwg5Yt8z68N91U3Exr550T+IEikq2ycoT+5Zfw+OM+UG7RIoEf9MYbvkDorrtg/Hg/pmQuIgmSdQm9oMB3Idp1V7j33gR9yC+/+OyVU0+F3XbzZlpt2ybow0REXNaVXJ591vPrc88lcLB8xhm+Ldy998INN0QwuV1EslFWJfTFi71ZYdu2cN55cX7zuXO9N3n16r4rxg47eLlFRCRJsqrk0rs3rFgBAwfGcVV9QQEMGODNs26/3Y81b65kLiJJlzUJffx4b5HSq1ccGxfOnAlHHw09evhGzT17xumNRUTKLysSel6ez2ipW7d4EL3NXn4ZmjWDadPgn//0HTEaNIjTm4uIlF9W1NAHDIApU+C113yZ/zYpaqbVogWcdpo309pjj7jEKSKyLTJ+hL5gge8T0bGjzyLcauvWwS23+AyWEGDffeHFF5XMRSRlZHxCv+4673312GPbcCN0/Hi/0dm3r89iUTMtEUlBGZ3QR4/21ik33eQD6nJbtcp3vmjTBtasgffegyFD1ExLRFJSxib09eu9Ne6++/rc862yYQO8+qq/0bRp0KFDXGMUEYmnjL0p+tBDPqvw3XehcuVyfOPSpb4w6NZbYZddvJlWjRoJi1NEJF4ycoT+ww++6v700+HEE8vxja+95guC7r23uJmWkrmIpImMTOg9e8J228HDD8f4DT//7Nn/jDN8C7jcXDXTEpG0k3Ell+HD/fHAA1CvXozfdNZZMHEi3HefT4upmHH/LCKSBSyEEMkH5+TkhNzc3Li+55o1XjGpVs17nm+2yeGPP3qNvHp1mDzZ9/bcf/+4xiMiEm9mNimEkFPWaxlVcunb1/P0wIGbSeYFBT4p/cADfcURwCGHKJmLSNrLmNrCzJlw//3Qtetmyt/ffAPdusG4cX63tFevpMYoIpJIGTFCD8Gniu+4oyf1Mg0d6s20Zszw3S1GjIC9905qnCIiiZQRI/T//MdXhQ4YALvvXurFggKf8nLYYXDmmT5B/Q8niYikv7S/KbpiBTRp4rMNP/sMKlQofGHtWt+ceeZMGDYsjjtaiIhEJ6Nvit5xByxcCIMGlUjmY8f6jc5+/aBWLdi4McoQRUSSIq0T+uTJvkr/ssu8osLKlV5Mb9vWk/j778PTT0OlSlGHKiKScGmb0AsKfBeiWrWgT5/Cgxs3whtvwDXXwFdfwfHHRxihiEhype1N0SFD4NNP4aXHl1Dz4Ud8b7lddvGpidWrRx2eiEjSxTRCN7MTzWymmc02s95lvG5m9mjh61PN7ND4h1psyRK44frAbU1e4f/ubgp//7tnd1AyF5GstcWEbmYVgAFAR6ApcI6ZNS11WkegUeGjOzAoznH+zn1XL+Dppadx9zdnYfXqeTOto45K5EeKiKS8WEboLYHZIYQ5IYQNwFCgU6lzOgHPBTcB2NnM9oxzrABMmACdXjyLkyu+56uIJkzwBUMiIlkulhp6HWBeiefzgVYxnFMH+LnkSWbWHR/BU79+/fLGCvgaoRdaD+CQx6uw/aGNt+o9REQyUSwJvawVOaVXI8VyDiGEwcBg8IVFMXz2H7RsCS3Ha0QuIlJaLCWX+UDJzuJ1gQVbcY6IiCRQLAl9ItDIzBqaWSXgbGB4qXOGA10LZ7scDiwPIfxc+o1ERCRxtlhyCSHkmVkPYCRQAXg2hDDdzC4rfP0JYARwEjAbWANclLiQRUSkLDEtLAohjMCTdsljT5T4OgBXxjc0EREpj7Rd+i8iIr+nhC4ikiGU0EVEMoQSuohIhohsxyIzWwT8uJXfviuwOI7hpANdc3bQNWeHbbnmvUMItct6IbKEvi3MLHdTWzBlKl1zdtA1Z4dEXbNKLiIiGUIJXUQkQ6RrQh8cdQAR0DVnB11zdkjINadlDV1ERP4oXUfoIiJSihK6iEiGSOmEnmqbUydDDNfcpfBap5rZeDNL+90+tnTNJc47zMzyzeyMZMaXCLFcs5kdY2aTzWy6mX2S7BjjLYb/bdcws7fMbErhNad111Yze9bMfjWzaZt4Pf75K4SQkg+8Ve93wD5AJWAK0LTUOScB7+I7Jh0OfBZ13Em45iOAmoVfd8yGay5x3od4188zoo47CT/nnYGvgfqFz3eLOu4kXPPNQL/Cr2sDS4FKUce+DdfcFjgUmLaJ1+Oev1J5hJ5Sm1MnyRavOYQwPoTwW+HTCfjuUOkslp8zwFXAa8CvyQwuQWK55nOBYSGEuQAhhHS/7liuOQDVzcyAanhCz0tumPETQhiDX8OmxD1/pXJC39TG0+U9J52U93ouxn/Dp7MtXrOZ1QFOBZ4gM8Tyc24M1DSzj81skpl1TVp0iRHLNT8OHIBvX/kV0DOEUJCc8CIR9/wV0wYXEYnb5tRpJObrMbNj8YTeJqERJV4s19wfuDGEkO+Dt7QXyzVXBFoAxwFVgE/NbEIIYVaig0uQWK65AzAZaAfsC7xvZmNDCCsSHFtU4p6/UjmhZ+Pm1DFdj5kdDDwNdAwhLElSbIkSyzXnAEMLk/muwElmlhdCeCMpEcZfrP/bXhxCWA2sNrMxQDMgXRN6LNd8EXBf8ALzbDP7HmgCfJ6cEJMu7vkrlUsu2bg59Rav2czqA8OA89N4tFbSFq85hNAwhNAghNAAeBW4Io2TOcT2v+03gaPMrKKZVQVaATOSHGc8xXLNc/G/SDCz3YH9gTlJjTK54p6/UnaEHrJwc+oYr/l2oBYwsHDEmhfSuFNdjNecUWK55hDCDDN7D5gKFABPhxDKnP6WDmL8Od8DDDGzr/ByxI0hhLRtq2tmLwHHALua2XzgDmB7SFz+0tJ/EZEMkcolFxERKQcldBGRDKGELiKSIZTQRUQyhBK6iEiGUEIXEckQSugiIhni/wGyNHX83OM6LAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "metricas_modelos(y_test, pred_clf_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc356098",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

