{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "from matplotlib import style\n",
    "# style.use('fivethirtyeight')\n",
    "style.use('seaborn')\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime as dt\n",
    "from datetime import date, timedelta, datetime"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reflect Tables into SQLAlchemy ORM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Python SQL toolkit and Object Relational Mapper\n",
    "import sqlalchemy\n",
    "from sqlalchemy.ext.automap import automap_base\n",
    "from sqlalchemy.orm import Session\n",
    "from sqlalchemy import create_engine, func, inspect"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create engine using the `hawaii.sqlite` database file\n",
    "\n",
    "engine = create_engine(\"sqlite:///Resources/hawaii.sqlite\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# reflect an existing database into a new model\n",
    "Base = automap_base()\n",
    "\n",
    "# reflect the tables\n",
    "Base.prepare(engine, reflect=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'measurement'"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# We can view all of the classes that automap found\n",
    "\n",
    "Base.classes.keys()[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save references to each table\n",
    "\n",
    "Measurement = Base.classes.measurement\n",
    "Station = Base.classes.station"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create our session (link) from Python to the DB\n",
    "\n",
    "session = Session(engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id INTEGER\n",
      "station TEXT\n",
      "date TEXT\n",
      "prcp FLOAT\n",
      "tobs FLOAT\n"
     ]
    }
   ],
   "source": [
    "# Show name of the rows and type in the Measurement table\n",
    "\n",
    "inspector = inspect(engine)\n",
    "measurement_columns = inspector.get_columns(\"Measurement\")\n",
    "\n",
    "for i in measurement_columns:\n",
    "    print(i['name'], i['type'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id INTEGER\n",
      "station TEXT\n",
      "name TEXT\n",
      "latitude FLOAT\n",
      "longitude FLOAT\n",
      "elevation FLOAT\n"
     ]
    }
   ],
   "source": [
    "# Show name of the rows and type in the Station table\n",
    "\n",
    "station_columns = inspector.get_columns(\"Station\")\n",
    "\n",
    "for i in station_columns:\n",
    "    print(i['name'], i['type'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Climate Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('2017-08-23',)]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Design a query to retrieve the last 12 months of precipitation data and plot the results\n",
    "\n",
    "# Retrieve the most recent date from the Measurement table\n",
    "\n",
    "ret_date = engine.execute('SELECT MAX(date) FROM Measurement').fetchall()\n",
    "ret_date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the date 1 year ago from the last data point in the database\n",
    "\n",
    "start_date = ('2016-08-24')\n",
    "end_date = ('2017-08-23')\n",
    "\n",
    "\n",
    "data = engine.execute(\"SELECT * FROM Measurement WHERE date >= ? AND date <= ?\\\n",
    "                        ORDER BY date ASC\", start_date, end_date).fetchall()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform a query to retrieve the date and precipitation scores\n",
    "\n",
    "date = []\n",
    "prep = []\n",
    "\n",
    "for i in data:\n",
    "    date.append(i[2])\n",
    "    prep.append(i[3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <th>Precipitation</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>0.08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>2.15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>2.28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>1.22</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Precipitation\n",
       "Date                     \n",
       "2016-08-24           0.08\n",
       "2016-08-24           2.15\n",
       "2016-08-24           2.28\n",
       "2016-08-24            NaN\n",
       "2016-08-24           1.22"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Save the query results as a Pandas DataFrame and set the index to the date column\n",
    "\n",
    "new_list = list(zip(date, prep))\n",
    "new_list\n",
    "\n",
    "data_df = pd.DataFrame(new_list, columns=[\"Date\", \"Precipitation\"])\n",
    "date_index = data_df.set_index(\"Date\")\n",
    "date_index.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
       "      <th>Precipitation</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>0.08</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>2.15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>2.28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>1.22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-08-24</th>\n",
       "      <td>2.15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Precipitation\n",
       "Date                     \n",
       "2016-08-24           0.08\n",
       "2016-08-24           2.15\n",
       "2016-08-24           2.28\n",
       "2016-08-24           1.22\n",
       "2016-08-24           2.15"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Sort the dataframe by date\n",
    "\n",
    "date_index.dropna(inplace=True)\n",
    "date_index.sort_index().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1gAAAI4CAYAAAB3HEhGAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdd5wU9f3H8fde4ygnzUONhRil2DvRaDRALIkSpCRWNPaGxoLyQxEUVETEiBIVC/aCIqDGqCDYAEWUpkhVersDjivccXd7O78/8I4rO7uzeztt9/V8PBK5LTOf/c53vjOf+X6/MwHDMAwBAAAAABotze0AAAAAACBZkGABAAAAQIKQYAEAAABAgpBgAQAAAECCkGABAAAAQIKQYAEAAABAgpBgAUAjrV+/Xocddph69uxZ87+//e1vmjhxYsLWMWbMGE2ZMiXiZ6ZPn64HHnhAkvT5559rzJgxUZdb+3O1v59oP/zwg7p16xb1c++8845ef/31mJb95ptv6tlnn43pO7XX8+STT2rYsGExfb++fv366eOPP47ru+vWrdPNN98c8TNLly7VaaedVue1r776Sr1791bPnj3Vq1cvzZw5M671N0bt+jNp0iRdd911jscAAF6T4XYAAJAMsrOz9d5779X8vWXLFp133nk68sgj1blz50Yv/1//+lfUz3Tv3l3du3eXtDuhKSwsjPqd2p+r/X23fP/99+rQoUNM37noooscWY9dNm7cqFWrVoV9LxgM6rXXXtNzzz2n0tLSmteLi4s1YMAAvfbaa+rQoYOWLl2qSy+9VJ9//rlatGjhVOiW6xkApBISLACwwT777KP27dtr9erV+umnnzRx4kSVlZWpRYsWevXVV/XOO+/ozTffVCgUUqtWrXTvvffqkEMO0c6dO/XAAw9o3rx5Sk9P15///GfddtttGjRokDp06KCrrrpKhx9+uK655hp99dVXKi0t1e23366zzjpLkyZN0ieffKIbb7xRb731lqqqqpSTk6PrrrtO9913n9asWaMdO3aoefPmevTRR1VcXFznc+3bt9cnn3yicePGafPmzbrvvvu0YcMGGYah888/X1dffbXWr1+vf/7znzrjjDO0cOFCFRUV6c4779SZZ57ZoAzeeOMNvfzyy2rRooU6duxY8/rWrVs1ZMgQbdu2Tfn5+dp///31+OOPa968eZoxY4ZmzZql7OxsnX322WE/17Zt2zrrefLJJ1VQUKAhQ4aoW7du6tWrl77++mtt2rRJPXv21K233lrn89OmTauzHkn65Zdf1K9fP+Xn52vvvffWY489pnbt2mnLli0aNmyYNm3apMrKSp177rm6/vrrI277Z555RtOnT9euXbtUVlamgQMH6swzz9TPP/+se+65RxUVFTIMQ3379tWFF16owYMHa8uWLbrqqqv0wgsv1FnWTz/9pGXLlmns2LG68sora16vrKzU0KFDa5LEQw89VIZhqKCgoEGCddRRR+mKK67Q7NmzVVpaqv79++vjjz/W8uXL1a5dOz3zzDNq1qyZvvvuOz3yyCMqKytTZmambr31Vp1++umaNGmSpk2bprS0NK1Zs0bZ2dkaOXKkSkpKGtSf/Px8XXvttdq0aZPS09M1evRoHXLIIZo6daqefvppBQIBpaen66677tJJJ50UsRwBwLcMAECjrFu3zjj22GPrvDZv3jzjpJNOMjZu3Gi8++67xkknnWQUFxcbhmEYc+bMMS6++GKjtLTUMAzD+Oqrr4xzzjnHMAzDeOihh4zbbrvNCAaDRnl5uXHJJZcY33zzjTFw4EDj+eefNwzDMDp27Gg8/fTThmEYxpIlS4wTTjjB2LZtm/Huu+8a1157rWEYhvHEE08Y999/v2EYhvHRRx8Zw4cPr4nt3nvvNYYNG9bgc7W/f8kllxjjx483DMMwioqKjB49ehj//e9/jXXr1hkdO3Y0ZsyYYRiGYXz88cfGn/70pwZl8tNPPxmnnHKKkZeXV7POrl27GoZhGC+99JIxbtw4wzAMIxQKGVdffbXxwgsvGIZh1PmdkT5XW+3f0LVrV+Phhx82DMMwNm/ebBx11FHG2rVrG3yn9nqeeOIJo1u3bsa2bdsMwzCMG264wRg7dqxhGIbRr18/Y/r06YZhGMauXbuMfv36GR9++GGD5V166aXGRx99ZKxfv97o16+fUVZWZhiGYfz3v/81zjvvPMMwDGPQoEE1vycvL8+49dZbjaqqKuObb74xzj333AbLrC1cHatt9OjRRu/evcO+17FjR+Pll182DMMwxo0bZxx33HHG5s2bjaqqKqNXr17G+++/b2zfvt045ZRTjAULFhiGYRjLly83unTpYqxdu9Z49913jRNOOMHYtGmTYRiGMWzYMOOuu+6qKbva9efEE080Vq9ebRiGYQwfPtwYNGiQYRiG0b17d2P+/PmGYeyu708++WTE3wsAfkYPFgAkwK5du9SzZ09JUlVVlVq3bq1Ro0Zpv/32kyR16tSppmfh888/15o1a3ThhRfWfL+oqEg7duzQ7NmzNWjQIKWnpys9PV2vvfaaJGny5Ml11nfppZdKkjp37qyOHTtq7ty5prGdc845OvDAA/Xqq69qzZo1+vbbb3XccceZfr60tFTz5s3T+PHjJUk5OTnq3bu3vvzySx1zzDHKzMzUGWecIUk6/PDDtWPHjgbL+Prrr3XqqacqNzdXknTBBRfUzBG6/PLL9d133+nFF1/U6tWrtWLFCh1zzDENlmH1c/VVD3PcZ5991LZtWxUWFurAAw+M+J1TTz1Vbdq0kbS7TLdv367S0lLNnTtXhYWFNfOMSktLtXTpUv31r38Nu5z9999fjzzyiD744AOtWbNGCxcu1M6dOyVJZ555pgYOHKhFixbplFNO0eDBg5WW1rip0MFgUA8//LC+/PJLvfTSS6afO/vssyVJBx10kDp27Kh99tlHknTAAQeosLBQixYt0kEHHVRTvh06dNDxxx+vb7/9VoFAQEcccYT23XdfSbu3+bRp08Ku5+ijj1b79u0lSYcddljN584991z1799fZ5xxhk499VRdc801jfrdAOBlJFgAkAD152DV16xZs5p/h0Ih9ezZU3feeWfN33l5eWrZsqUyMjIUCARqPrtp06aaYWy1paen11le7b/re+ONN/T222/rkksuUY8ePdSqVSutX7/e9POhUEiGYTR4LRgMSpIyMzNrEoPasdZXexm14xs1apQWLVqkPn366Pe//72CwWCD9cXyufqaNGlS8+9AIGDpOxkZew6H1d+pLoe33npLTZs2lSRt3769zvLrW7x4sW688Ub985//1KmnnqqTTjpJ999/vySpa9eu+uSTTzR79mx9/fXX+s9//qNJkyZFjc1MYWGhbrnlFhmGoQkTJqh169amn83MzAz772pVVVUNtqVhGAoGg8rMzKxTByOVabhylKTbbrtNffr00axZszRp0iSNHz8+oTeBAQAv4S6CAOCw0047TR9++KHy8vIk7b4L3uWXXy5JOuWUUzR58mSFQiFVVFTolltuCds7VX1HwcWLF2vVqlUN5rOkp6fXJEQzZ85Ur1699Pe//10HH3ywZsyYoaqqqgafq9aiRQsdc8wxNXfZKy4u1pQpU/SHP/zB8m889dRTNWvWLG3evFlS3R64mTNn6vLLL9f555+vtm3bavbs2WHjifS5xgr3u+tr0aKFjj32WL344ouSdvcyXnTRRZo+fbrpd+bOnasjjzxSV1xxhbp06aLp06fXxHzHHXfof//7n84991wNHTpULVq00Nq1a5Wenq7KysqY4q+qqtK1116rAw44QOPHj4+YXFlx7LHH6pdfftGiRYskSStWrNDcuXPVpUuXiN+zUo7BYFDdunVTWVmZLrroIg0dOlTLli1TRUVFo2IGAK+iBwsAHHbaaafpmmuu0ZVXXqlAIKAWLVpo7NixCgQC6t+/vx588EH17NlTVVVV+utf/6qzzjpLM2bMqLOMefPm6e2331YoFNK///1vtWzZss77J598sgYMGKDhw4fryiuv1JAhQ2p6DI499lgtX768weeOOOKImu8/+uijGjZsmCZNmqSKigr16NFDvXv31oYNGyz9xk6dOunOO+/U5ZdfrubNm+voo4+uee+mm27SI488ojFjxigzM1PHH3+81q5dK0k6/fTT9fDDD0f9XGPVXk8kjz76qIYPH64ePXqooqJC5513nv72t7+Zfv68887T1KlT9Ze//EWhUEhdu3ZVYWGhSkpKdOONN+qee+7RhAkTam5gctJJJ6mwsFBNmjRR37599c4770TsFaz20UcfacGCBSotLVWfPn1qXn/kkUfUqVMna4VQS5s2bTRmzBgNHz5cu3btUiAQ0IgRI3TwwQdr/vz5pt8zqz+1ZWRk6O6779aAAQNqemgfeughZWVlxRwnAPhBwLAydgIA4BmdOnXS119/XTNnCAAAeAdDBAEAAAAgQejBAgAAAIAEoQcLAAAAABKEBAsAAAAAEsS2uwhOmjSp5ra85eXlWrJkiWbNmqW99tor7Ofz84vtCiVurVs3U0FBqdthpAzK21mUt7Mob2dR3s6ivJ1FeTuL8naWn8o7Nzcn7Ou2JVi9e/dW7969JUn333+/+vTpY5pceVVGhvmDO5F4lLezKG9nUd7OorydRXk7i/J2FuXtrGQob9uHCP7www9auXKlLrjgArtXBQAAAACusv0ugv3799ell16qk08+OeLngsGqpMhYAQAAAKQu24YISlJRUZF++eWXqMmVJE+O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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Use Pandas Plotting with Matplotlib to plot the data\n",
    "\n",
    "date_index.plot(title=\"Precipitation data in the last 12 months\", figsize=(12,8))\n",
    "plt.ylabel(\"Precipitation (inches)\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
       "      <th>Precipitation</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2015.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.460</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.020</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>6.700</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Precipitation\n",
       "count       2015.000\n",
       "mean           0.176\n",
       "std            0.460\n",
       "min            0.000\n",
       "25%            0.000\n",
       "50%            0.020\n",
       "75%            0.130\n",
       "max            6.700"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Use Pandas to calcualte the summary statistics for the precipitation data\n",
    "\n",
    "date_index.describe().round(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The number of stations in this dataset are 9\n"
     ]
    }
   ],
   "source": [
    "# Design a query to show how many stations are available in this dataset?\n",
    "\n",
    "stats_num = session.query(Station).distinct(Station.station).count()\n",
    "print(\"The number of stations in this dataset are \" + str(stats_num))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The most active stations, in descending order, are as follows: \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[('USC00519281', 2772),\n",
       " ('USC00519397', 2724),\n",
       " ('USC00513117', 2709),\n",
       " ('USC00519523', 2669),\n",
       " ('USC00516128', 2612),\n",
       " ('USC00514830', 2202),\n",
       " ('USC00511918', 1979),\n",
       " ('USC00517948', 1372),\n",
       " ('USC00518838', 511)]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# What are the most active stations? (i.e. what stations have the most rows)?\n",
    "# List the stations and the counts in descending order.\n",
    "\n",
    "stats_count = session.query(Measurement.station, func.count(Measurement.station)).\\\n",
    "                group_by(Measurement.station).order_by(func.count(Measurement.station).\\\n",
    "                                                       desc()).all()\n",
    "\n",
    "print(\"The most active stations, in descending order, are as follows: \" )\n",
    "stats_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The lowest temperature recorded in the most active station, USC00519281, is 54.0\n",
      "The highest temperature recorded in the most active station, USC00519281, is 85.0\n",
      "The average temperature recorded in the most active station, USC00519281, is 71.66\n"
     ]
    }
   ],
   "source": [
    "# Using the station id from the previous query, calculate the lowest temperature recorded, \n",
    "# highest temperature recorded, and average temperature of the most active station?\n",
    "\n",
    "stats_min_temp = session.query(func.min(Measurement.tobs)).filter(Measurement.station == stats_count[0][0]).all()\n",
    "stats_max_temp = session.query(func.max(Measurement.tobs)).filter(Measurement.station == stats_count[0][0]).all()\n",
    "stats_avg_temp = session.query(func.avg(Measurement.tobs)).filter(Measurement.station == stats_count[0][0]).all()\n",
    "\n",
    "print(f'{\"The lowest temperature recorded in the most active station, \" + str(stats_count[0][0]) + \", is \" + str(stats_min_temp[0][0])}')\n",
    "print(f'{\"The highest temperature recorded in the most active station, \" + str(stats_count[0][0]) + \", is \" + str(stats_max_temp[0][0])}')\n",
    "print(f'{\"The average temperature recorded in the most active station, \" + str(stats_count[0][0]) + \", is \" + str(round((stats_avg_temp[0][0]), 2))}')\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Station USC00519281 has the highest number of temperature observations\n"
     ]
    }
   ],
   "source": [
    "# Choose the station with the highest number of temperature observations.\n",
    "\n",
    "stats_hi = stats_count[0][0]\n",
    "print(\"Station \" + str(stats_hi) + \" has the highest number of temperature observations\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Query the last 12 months of temperature observation data for this station \n",
    "\n",
    "stats_hi = ('USC00519281')\n",
    "start_date = ('2016-08-24')\n",
    "end_date = ('2017-08-23')\n",
    "\n",
    "\n",
    "data_one = engine.execute(\"SELECT * FROM Measurement WHERE station = ? AND \\\n",
    "                        date >= ? AND date <= ? ORDER BY date DESC\", stats_hi, start_date, end_date).fetchall()\n",
    "data_one\n",
    "\n",
    "\n",
    "    # Create an empty list to store the values of temperature in the station for the last 12 months\n",
    "stats_temp = []\n",
    "\n",
    "for i in data_one:\n",
    "    stats_temp.append(i[4])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1gAAAI4CAYAAAB3HEhGAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3de4Df853v8dcvmZAmkgoJVaQV5RCUbSJaTaZ1a9wiriUUKVvl1CV1SyghdYlEKbLqUos9UbdVZa09tURVJZGqHkWWVh3ENQ2CJCKZTL7nD8esVMS0/fxmTPp4/DXzu33e+f2+c3nm+/19p1ZVVRUAAAD+Zp3aewAAAICVhcACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBbwd+Pss8/O8OHDM3z48Gy++eYZOnRoy+fvvPNOe4/3kaqqysiRI/PWW2/VdZ1p06blG9/4Rr72ta9ljz32yOGHH56HH3645foTTzwx1157bV1n+Es88sgjOfPMM5Mkv/vd7zJq1Ki6rtfc3Jzhw4dn/vz5rb7Pm2++mZEjRyZJlixZkv/xP/7H3/Q6Tps2LcOHD/+r73/JJZfkF7/4xYdev3Tp0g+8zgsXLszo0aMzbNiw7Lbbbvne976XRYsW/dUz/DX+/GugsbExTzzxRJvOAPBRGtp7AIC2ctppp7V8vP322+cHP/hBtthii3ac6C/T3Nyc6dOn13WN//zP/8wPfvCDnH/++dlyyy2TJL/97W9z/PHH5+yzz87gwYPruv5f46mnnsqf/vSnJMmWW26Ziy66qK7rde7cObfffvtfdJ833ngjjz/+eJ0m+stNnz49/fv3X+51Tz31VMaNG5fHH388m2++ecvll156aTp16pTbb789VVXl+OOPz49//OMcffTRbTV2m3wNAPytBBbA//fUU0/lnHPOyVtvvZXm5uaMHDkye+21V6ZNm5ZJkyalT58+ee6559KtW7f84z/+YyZPnpxnn302u+yyS0aPHp1p06blkksuyVprrZVnnnkm3bp1y/jx49OvX78sXrw4EydOzMMPP5zm5uZsttlm+d73vpfVVlstjY2NGTBgQJ588smcdNJJWbp0aa666qosXrw4r7/+evbZZ58cc8wxOeWUU5IkBx10UK666qrst99+ueKKK7Lpppsmefd/86+44op069Yt3/zmN9O3b9+8/PLLuf766/PMM8/kggsuyDvvvJNOnTrl2GOPzVe+8pUPPAcTJ07MuHHjWuIqSb7whS9k9OjRmThxYktg/eY3v8mdd96ZBQsWZMiQITnppJPS0NCQH/7wh7n33nvTpUuX9OrVKxMmTEjv3r1X+NxOnDgxq666at55552sv/762XrrrXPooYcmSSZPnpxHHnkkEydOzLnnnpvHHnusZc/Rueeem969e+fSSy/NvHnz8r3vfS+77bZbJkyYkNtvvz1vvfVWxo0bl9///vdJku222y7HHXdckmTgwIEZOXJkHnzwwfzpT3/Kt7/97ey///6ZPXt2Ro8enTfffDPJuyF+zDHHLPMcLVmyJJtttlkeeuih3HXXXbnvvvuydOnSPP/88/nEJz6RCRMmpF+/fsvc55RTTsmCBQsyfPjw/PSnP02SXHTRRXnkkUfyxhtv5Fvf+lZGjBiRJLnpppty0003ZenSpVljjTVy+umnZ4MNNvjQ7fbpp5/OWWedlYULF2b27NnZbLPN8sMf/jCrrLLKcl+P//iP/8iTTz6Zc889N7VaLTvssMMyj/eTn/wkBx54YO69995lLh80aFA+85nPpFOndw9+2XTTTTNr1qwPzPPDH/4wr7zySl566aXMmTMnn//85zNgwIDcfvvtefHFFzN69OjsuuuuWbx4ccaPH59f//rX6dSpU7baaquMGTMm3bt3T2NjY77+9a9n2rRpefnll7P33nsv92sgSa6//vrMnDkzr7/+evbee+8ce+yxmT9/fk455ZTMmjUrnTp1yhZbbJFx48alVqt96PMIUEwF8Hdou+22qx599NGWzxcvXlztsssu1RNPPFFVVVW9+eab1dChQ6tHH320mjp1atW/f/+W60aOHFmNGDGiWrx4cfXqq69Wm266afXqq69WU6dOrTbZZJPq4YcfrqqqqiZPnlztt99+VVVV1UUXXVSdf/751dKlS6uqqqoJEyZUZ511VlVVVTVkyJDq8ssvr6qqqpqbm6sDDzywmjVrVlVVVfXSSy9Vm2yySfXGG29UTU1N1cYbb1y9+eabLff7r//6r5Z/w3ufP/vss9XGG29c/fa3v62qqqpef/316mtf+1r14osvVlVVVS+//HI1ZMiQ6uWXX17mOZkzZ0618cYbV++8884Hnq833nij2njjjat58+ZVJ5xwQrXvvvtWb7/9drVo0aJqxIgR1U033VTNmjWr2nrrratFixZVVVVVV155ZTVlypSPfG433XTTllkeeOCBao899mhZd6+99qoefPDB6qGHHqpGjRpVNTc3V1VVVZdeemn1P//n/6yqqqpuvvnm6qijjqqqqqqmTp3acv/jjz++Gj9+fFVVVfXOO+9Uhx56aHXVVVe1PI/XX399VVVV9cgjj1RbbLFFtXjx4uriiy+uxo0bV1VVVc2fP7869thjq3nz5i3zXLz/dbj55purrbfeunrllVeqqqqqsWPHVqeccsoHnr9nn322GjBgwDL3v/baa6uqqqrf/e531ec///mqubm5mjZtWvWNb3yjWrhwYVVVVXXfffdVu++++wce7/3/znPPPbf693//96qq/ns7vvvuuz/09aiqqjrggAOqu++++wOP+34nnHBCdc011yz3ulmzZlXbbrtt9ctf/vID11144YXVjjvuWM2bN696++23qy984QvVxIkTq6qqqp///OfVLrvs0nK74447rmpqaqqWLFlSnXzyyS3P/ZAhQ6rzzz+/qqp3vwY222yz6qWXXlru18A555xTVVVVvfLKK9Vmm21WzZ49u7rllluqI444ouX5PuWUU1q+pgDqzR4sgLy7F+D555/P6NGjWy5bvHhxnnjiiay33nrp27dvNtlkkyTJ+uuvn969e6dLly5Zc801061bt7zxxhtJks022yxf+MIXkiT77bdfzj777MybNy/33Xdf3n777fzqV79KkjQ1NWWttdZqWWvAgAFJkk6dOuWKK67Ifffdl9tvvz1//OMfU1VV3nnnnXTv3r3V/54uXbosc4jfnDlzctRRR7Vc36lTp/zhD3/Ipz71qQ/ct6mpKauuuuoyly1evDhJWvYA7LnnnvnEJz6RJNljjz0yderU7L333tlwww2z9957p7GxMY2NjfniF7+YJ598coXP7brrrtsyx7bbbpt58+bliSeeSK1Wy4IFCzJo0KDUarX06tUrN954Y2bNmpVf//rX+eQnP7nC5+CBBx7ILbfckiRZddVVs//+++fGG29s2Tv23p6b/v37Z9GiRXnnnXfS2NiYb3/723nhhRey7bbb5uSTT85qq622wnW22GKLrL322kneff1/+ctfrvD27xk2bFiSd/cEvfPOO1mwYEHuu+++PPPMM9l///1bbjd37tzMmzcvPXr0WO7jnHzyyZk6dWquvPLKPPvss3nttdfy9ttvZ5111lnu6/G3evTRR3PMMcdk5MiRaWxsXO5ttt1225bnrU+fPhkyZEiSpG/fvi17B++///6MHj06DQ3v/ipy0EEH5fjjj295jPden3XWWSe9evXKm2++mT59+nxgrd133z1Jsvbaa6dXr155/fXXs/XWW+fiiy/OIYcckm233TaHH3541l9//b/53w7QGgILIO++qX/11Vdf5r01c+bMSc+ePfPwww9nlVVWWeb27/1S+Ofef/nSpUuTvBszzc3NGTt2bL785S8nSebPn5+mpqaW274XT/Pnz89ee+2VoUOHZsCAAdlnn31y9913p6qqD6xVq9WWufz9j9e1a9eWQ7mWLl2ajTfeODfeeGPL9bNnz84aa6yxzOP17t07ffv2zUMPPZTttttumetmzJiRjTfeuGXO9x77vcdvaGhIQ0NDrr/++jz66KOZPn16zj777Gy//fbZeeedV/jcduvWbZl/07777pvbbrstVVVln332Sa1Wyz333JOJEydm5MiR2XHHHfPZz342P//5z5f7Grynubl5mUPCqqrKkiVLlnmO3lvzveu32mqrTJkyJdOnT8+DDz6YfffdN1dffXXLYZjL8/4Y/fPXZEXe21bev35zc3P22WeffPe73235N8yZM+dD4ypJRo0alVqtlp133jnbb799XnjhhVRV9aGvx/sj5i/1b//2bzn77LNz5plnZtddd/3Q27Xm66W1r0+y4ue1S5cuH7hd3759c/fdd2fGjBl58MEHc+ihh+acc85Z7mGxAKU5iyBAks997nPp1KlT7rzzziTJiy++mN133z1PPvnkX/Q4jz/+eJ566qkk776XZuutt0737t0zePDgTJ48OU1NTWlubs6pp5663JMxPPPMM1m4cGGOO+64bLfddpk+fXqWLFmS5ubmdO7cObVareWX0DXWWKPlxAnTpk3L66+/vtyZ/uEf/iFPP/10y5kAZ86cmaFDh+a11177wG3HjBmTs88+O48++mjLZQ8//HAmTpyYE088seWyO++8M4sXL84777yT22+/PY2NjZk5c2b22GOPbLTRRjnyyCNzyCGH5LHHHvuLn9t99tkn99xzT/7zP/8ze++9d8u/b4cddsiBBx6YzTffPPfcc0+am5uTvHvSiff/Yv6ewYMH57rr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AKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIUILAAAgEIEFgAAQCECCwAAoBCBBQAAUIjAAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAppaO8BAAD42x123r3tPUKbu3rM9u09AnyAPVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIUILAAAgEIEFgAAQCEN9XzwPffcMz169EiSrLfeetl///1zzjnnpHPnzhk8eHCOPvroei4PAADQpuoWWIsWLUqSTJ48ueWy4cOHZ9KkSVl//fVzxBFHZObMmdlss83qNQIAAECbqtshgk8++WQWLlyYww47LIccckgeeuihLF68OH379k2tVsvgwYMzffr0ei0PAADQ5uq2B6tr1645/PDDs99+++XZZ5/Nt771rfTs2bPl+u7du+f5559f4WP06tUtDQ2d6zVih9SnT4/2HoGVnG2MtmA7A0po7+8l7b0+H091C6wNNtggn/nMZ1Kr1bLBBhukR48eeeONN1quX7BgwTLBtTxz575dr/E6pD59emTOnHntPQYrMdsYbcF2BpTSnt9LfC/jwwK7bocI3nLLLTnvvPOSJLNnz87ChQvTrVu3zJo1K1VV5YEHHsjAgQPrtTwAAECbq9serH333TennHJKRowYkVqtlnPPPTedOnXKiSeemObm5gwePDhbbrllvZYHAABoc3ULrFVWWSUXXHDBBy6/+eab67UkAABAu/KHhgEAAAoRWAAAAIUILAAAgEIEFgAAQCECCwAAoBCBBQAAUIjAAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIUILAAAgEIEFgAAQCECCwAAoBCBBQAAUIjAAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIUILAAAgEIEFgAAQCECCwAAoBCBBQAAUIjAAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgkIb2HgCAtnfYefe29wht7uox27f3CAD8HbAHCwAAoBCBBQAAUIjAAgAAKERgAQAAFFLXwHrttdfyla98JU8//XSee+65jBgxIgceeGDOOOOMLF26tJ5LAwAAtLm6BVZTU1PGjh2brl27JknGjx+fUaNG5frrr09VVZkyZUq9lgYAAGgXdTtN+4QJE3LAAQfkyiuvTJLMnDkzgwYNSpI0NjZm6tSp2WmnnVb4GL16dUtDQ+d6jdgh9enTo71HYCVnG2Nl9fd4anpY2bX3z6z2Xp+Pp7oE1q233po11lgjQ4YMaQmsqqpSq9WSJN27d8+8efM+8nHmzn27HuN1WH369MicOR/9vMFfyzYGQEfSnj+z/MzkwwK7LoH105/+NLVaLdOnT88TTzyR0aNH5/XXX2+5fsGCBenZs2c9lgYAAGg3dQmsn/zkJy0fH3zwwTnzzDNz/vnnZ8aMGdlmm21y//3354tf/GI9lgYAAGg3bXaa9tGjR2fSpEnZf//909TUlKFDh7bV0gAAAG2ibie5eM/kyZNbPr7uuuvqvRwAAEC78YeGAQAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKCQVgXWuHHj8uijj9Z7FgAAgA6toTU3+vznP58LLrggr7/+eoYPH57hw4enT58+9Z4NAACgQ2nVHqy99tor//Iv/5Irr7wyVVXlgAMOyLe//e3cc8899Z4PAACgw2j1e7Cef/753HrrrfnZz36Wz3zmM9lpp53yv//3/87JJ59cz/kAAAA6jFYdIjhixIi8+uqr2XPPPXPVVVfl05/+dJJkzz33TGNjY10HBAAA6ChaFVjHHntsBg4cmC5dumTJkiV5++23061btzQ0NGTatGn1nhEAAKBDaNUhgnPnzs3ee++dJHnppZey8847e/8VAADAn2lVYF122WW55pprkiR9+/bNz372s0yaNKmugwEAAHQ0rQqspqam9O7du+XzNddcM1VV1W0oAACAjqhV78EaMGBAjj/++AwbNiy1Wi3/8R//ka222qreswEAAHQorQqsM844I//rf/2v3HTTTWloaMjAgQNz4IEH1ns2AACADqVVgbXKKqvkgAMOyG677dZyaOCrr77acrp2AAAAWhlYl19+ea688sqsvvrqqdVqqaoqtVotU6ZMqfd8AAAAHUarAuuWW27JPffckzXWWKPe8wAAAHRYrTqL4DrrrJNPfvKT9Z4FAACgQ2vVHqzPfvazOfDAA7PNNttklVVWabn86KOPrttgAAAAHU2rAmvttdfO2muvXe9ZAAAAOrRWBZY9VQAAAB+tVYG1ySabpFarLXPZWmutlV/+8pd1GQoAAKAjalVgPfnkky0fNzU15Z577skjjzxSt6EAAAA6oladRfD9unTpkl122SUPPvhgPeYBAADosFq1B+u2225r+biqqjz11FNpaGjVXQEAAP5utKqSZsyYscznvXr1ykUXXVSXgQAAADqqVgXW+PHj6z0HAABAh9eqwNp+++0/cBbB5N3DBWu1WqZMmVJ8MAAAgI6mVYE1bNiwdOnSJV//+tfT0NCQO+64I4899li++93v1ns+AACADqNVgfWrX/0qt956a8vnhx56aPbee++su+66dRsMAACgo2n1adqnTZvW8vEvfvGLdO/evS4DAQAAdFSt2oP1/e9/P6NHj86rr76aJOnXr18mTJhQ18EAAAA6mlYF1uabb54777wzr7/+erp27Zpu3brVey4AAIAOp1WHCL744ov55je/mQMOOCALFizIIYcckhdeeKHeswEAAHQorQqssWPH5vDDD0+3bt3Su3fv7L777hk9enS9ZwMAAOhQWhVYc+fOzeDBg5MktVotX//61zN//vy6DgYAANDRtCqwunbtmldeeaXljw3/5je/ySqrrFLXwQAAADqaVp3k4pRTTsm3v/3tzJo1K8OHD8+bb76Ziy++uN6zAQAAdCitCqzXXnstt9xyS5599tk0NzenX79+9mABAAD8mVYdInj++eenS5cu2WijjbLJJpuIKwAAgOVo1R6s9ddfP6ecckq23HLLdO3ateXyPffcs26DAQAAdDQrDKzZs2dn7bXXTq9evZIkv/vd75a5XmABAAD8txUG1pFHHpmf/exnGT9+fK6++uocdthhbTUXAABAh7PC92BVVdXy8R133FH3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AG2uboH14x//OKeddloWLVqUJBk/fnxGjRqV66+/PlVVZcqUKfVaGgAAoF3ULbD69u2bSZMmtXw+c+bMDBo0KEnS2NiYadOm1WtpAACAdtFQrwceOnRoXnjhhZbPq6pKrVZLknTv3j3z5s37yMfo1atbGho612vEDqlPnx7tPQIAAO3E74Iff3ULrD/XqdN/7yxbsGBBevbs+ZH3mTv37XqO1OH06dMjc+Z8dJgCALBy8rvgx8eHxW6bnUWwf//+mTFjRpLk/vvvz8CBA9tqaQAAgDbRZoE1evToTJo0Kfvvv3+ampoydOjQtloaAACgTdT1EMH11lsvN998c5Jkgw02yHXXXVfP5QAAANqVPzQMAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAACikob0HAAAAWuew8+5t7xHaxdVjtm/vEVrNHiwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABTiNO20yt/rKUEBAOAvYQ8WAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAppaO8BOprDzru3vUcAAAA+puzBAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAoRGABAAAUIrAAAAAKEVgAAACFCCwAAIBCBBYAAEAhAgsAAKAQgQUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIUILAAAgEIEFgAAQCECCwAAoBCBBQAAUIjAAgAAKERgAQAAFCKwAAAAChFYAAAAhQgsAACAQgQWAABAIQILAACgEIEFAABQiMACAAAopKEtF1u6dGnOPPPM/P73v88qq6ySs88+O5/5zGfacgQAAIC6adM9WPfcc08WL16cm266KSeccELOO++8tlweAACgrto0sB5++OEMGTIkSbLVVlvl8ccfb8vlAQAA6qpNDxGcP39+VltttZbPO3funCVLlqShYflj9OnTo61Ga7U7Lhje3iMAAAAfU226B2u11VbLggULWj5funTph8YVAABAR9OmgfWFL3wh999/f5LkkUceycYbb9yWywMAANRVraqqqq0We+8sgn/4wx9SVVXOPffcbLjhhm21PAAAQF21aWABAACszPyhYQAAgEIEFgAAQCECCwAAoBDnSP8Yu+KKK3LvvfemqakpI0aMyKBBgzJmzJjUarVstNFGOeOMM9Kpk0bmr/fn21j//v1z5JFH5rOf/WySZMSIEdl1113bd0g6tFtvvTU/+9nPkiSLFi3KE088kcmTJ+ecc85J586dM3jw4Bx99NHtPCUd2fK2sQsuuCATJ07MOuuskyQ55phjMmjQoPYckw6uqakpY8aMyYsvvphOnTrlrLPOSkNDg9/LWC4nufiYmjFjRq655pr86Ec/ysKFC3P11Vdn5syZ+eY3v5ltttkmY8eOzZAhQ7LTTju196h0UMvbxj71qU9l3rx5Oeyww9p7PFZC48aNyyabbJLrr78+kyZNyvrrr58jjjgio0aNymabbdbe47ESeG8be+mll9K/f/8MHTq0vUdiJXHPPffkjjvuyMUXX5ypU6fmxhtvTFNTk9/LWC6Z/TH1wAMPZOONN853vvOdHHnkkfnqV7+amTNntvwPXGNjY6ZNm9bOU9KRLW8be/zxx3PffffloIMOyqmnnpr58+e395isJB577LH88Y9/zG677ZbFixenb9++qdVqGTx4cKZPn97e47ESeG8b23///TNz5sz89Kc/zYEHHpjzzjsvS5Ysae/x6OA22GCDNDc3Z+nSpZk/f34aGhr8XsaHcojgx9TcuXPz0ksv5fLLL88LL7yQo446KlVVpVarJUm6d++eefPmtfOUdGTL28aOOOKI7Lffftl8881z2WWX5dJLL83o0aPbe1RWAldccUW+853vZP78+VlttdVaLu/evXuef/75dpyMlcV721iSfPnLX86OO+6Y9dZbL2eccUZuvPHGfOMb32jnCenIunXrlhdffDG77LJL5s6dm8svvzwPPfSQ38tYLoH1MbX66qunX79+WWWVVdKvX7+suuqqeeWVV1quX7BgQXr27NmOE9LRLW8b++pXv5o111wzSbLTTjvlrLPOaucpWRm89dZb+b//9//mi1/8YubPn58FCxa0XOd7GSW8fxtLkn322adlu9phhx1y1113ted4rASuvfbaDB48OCeccEJefvnlHHrooWlqamq53vcy3s8hgh9TAwYMyK9+9atUVZXZs2dn4cKF+dKXvpQZM2YkSe6///4MHDiwnaekI1veNnbEEUfk0UcfTZJMnz7d+2Io4qGHHsq2226bJFlttdXSpUuXzJo1K1VV5YEHHvC9jL/Z+7exqqqyxx57tPynpO9llNCzZ8/06NEjSfLJT34yS5YsSf/+/f1exnI5ycXH2MSJEzNjxoxUVZXvfve7WW+99XL66aenqakp/fr1y9lnn53OnTu395h0YH++ja2xxho566yz0qVLl/Tu3TtnnXXWModzwV/jqquuSkNDQ0aOHJkkeeSRR3Luueemubk5gwcPzne/+932HZAO78+3sQceeCAXXXRRunbtmg033DCnnXZaunTp0r5D0qEtWLAgp556aubMmZOmpqYccsgh2Xzzzf1exnIJLAAAgEIcIggAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIX4Q8MAtJlx48blt7/9bZqamjJr1qxsuOGGSZJDDjkk++yzTztP90EnnXRSTjzxxKy99trtPQoAHYTTtAPQ5l544YUccsghuffee9t7lBVqbGzMzTffnE996lPtPQoAHYQ9WAC0u/nz5+f73/9+/vjHP2bp0qU54ogjsuuuu+Zf//VfM3Xq1Lz22muZM2dODjzwwDz33HP59a9/nTXXXDNXXnllXn755Rx33HHp27dvnn766ay33no5//zz07Nnz9x3332ZNGlSmpub07dv33z/+9/P6quvnsbGxgwYMCBPPPFEbrjhhvzzP/9zZsyYkbfeeitrrbVWLr744tx00015/fXXc/jhh+eGG27I7rvv3hJb06ZNy5VXXplrr702I0aMSO/evfOHP/whl1xySV5++eXlrgnA3wfvwQKg3V166aXZcsstc+utt2by5Mm59NJL8+KLLyZJHnvssVx22WW58sorc84552THHXfMHXfckcWLF2fatGlJkieffDKHHHJI7rzzzvTt2zeXXnppXn311Vx00UW55pprctttt2WbbbbJhRde2LLmV7/61fz85z/P3Llz8/zzz+emm27KXXfdlbXWWit33HFHjjrqqKyxxhr553/+5/Ts2XOF82+66aa56667suaaa65wTQBWfvZgAdDupk2blqamptx8881JkoULF+aPf/xjkmTAgAFZbbXV0rVr19RqtWyzzTZJkk9/+tN58803kyQbbrhhBg4cmCTZc889c+qpp2brrbfOSy+9lIMPPjhJ0tzcnDXXXLNlzS233DJJ0q9fv5x44om5+eab8+yzz+bRRx/N5z73ub9o/vce65FHHlnhmgCs/AQWAO1u6dKlufDCC7PJJpskSV599dV88pOfzG233ZYuXbq03K5Tp07p1OmDB1907tx5mcfq3LlzmpubM2jQoPzTP/1TkmTRokVZsGBBy+26du2aJHn00Udz0kkn5Zvf/GZ23nnnVFWV5b09uVartVze1NS0zHWrrrpqknzkmgCs/BwiCEC722abbXLDDTckSWbPnp1hw4blT3/6U6vv//TTT+f3v/99kuTWW29NY2Njttpqq/zmN7/JrFmzkiSXXHJJLrjggg/cd8aMGfnSl76UAw44IH379s0vf/nLLF26NEnS0NCQJUuWJEl69eqVp556KkkyZcqU5c7R2jUBWHnZgwVAuzvuuONy5plnZtiwYWlubs6YMWOy7rrrtvr+vXr1yg9/+MPMmjUrm266aU4++eR84hOfyNlnn52jjz46S5cuzac//elMnDjxA/fdfffdc/TRR2fYsGFJki222CIvvPBCknffp3X44YfnmmuuybHHHptx48Zl9dVXz+DBg1si6v3WXnvtVq0JwMrLadoB6NCee+65/OM//mPuvvvu9h4FABwiCAAAUIo9WAAAAIXYgwUAAFCIwAIAAChEYAEAABQisAAAAAoRWAAAAIX8P6bN3Y7wanQNAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the results as a histogram\n",
    "\n",
    "plt.figure(figsize=(12,8))\n",
    "plt.hist(stats_temp, bins=12)\n",
    "plt.xlabel('Temperature')\n",
    "plt.ylabel('Frequncy')\n",
    "plt.title('Temperature Observations in the last 12 months')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bonus Challenge Assignment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[(62.0, 69.57142857142857, 74.0)]\n"
     ]
    }
   ],
   "source": [
    "# This function called `calc_temps` will accept start date and end date in the format '%Y-%m-%d' \n",
    "# and return the minimum, average, and maximum temperatures for that range of dates\n",
    "\n",
    "\n",
    "def calc_temps(start_date, end_date):   \n",
    "    \"\"\"TMIN, TAVG, and TMAX for a list of dates.\n",
    "    \n",
    "    Args:\n",
    "        start_date (string): A date string in the format %Y-%m-%d\n",
    "        end_date (string): A date string in the format %Y-%m-%d\n",
    "        \n",
    "    Returns:\n",
    "        TMIN, TAVE, and TMAX\n",
    "    \"\"\"\n",
    "    \n",
    "    return session.query(func.min(Measurement.tobs), func.avg(Measurement.tobs), func.max(Measurement.tobs)).\\\n",
    "        filter(Measurement.date >= start_date).filter(Measurement.date <= end_date).all()\n",
    "\n",
    "\n",
    "\n",
    "# function usage example\n",
    "print(calc_temps('2012-02-28', '2012-03-05'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[(60.0, 70.7156862745098, 80.0)]\n"
     ]
    }
   ],
   "source": [
    "# Use your previous function `calc_temps` to calculate the tmin, tavg, and tmax \n",
    "# for your trip using the previous year's data for those same dates.\n",
    "\n",
    "\n",
    "temp_values = calc_temps('2016-02-14', '2016-02-29')\n",
    "print(calc_temps('2016-02-14', '2016-02-29'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the results from your previous query as a bar chart. \n",
    "# Use \"Trip Avg Temp\" as your Title\n",
    "# Use the average temperature for the y value\n",
    "# Use the peak-to-peak (tmax-tmin) value as the y error bar (yerr)\n",
    "\n",
    "\n",
    "style.use('fivethirtyeight')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(4, 8))\n",
    "y = temp_values[0][1]\n",
    "x_pos = 1\n",
    "error = temp_values[0][2] - temp_values[0][1]\n",
    "bar = ax.bar(x_pos, y, yerr=error, alpha=0.5, color='green', align='center')\n",
    "ax.set(xticks=range(x_pos), xticklabels=\"a\", title=\"Trip Average Temperature\", ylabel=\"Average Temperature (F)\")\n",
    "ax.margins(.2, .2)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('USC00514830',\n",
       "  'KUALOA RANCH HEADQUARTERS 886.9, HI US',\n",
       "  21.5213,\n",
       "  -157.8374,\n",
       "  7.0,\n",
       "  0.98),\n",
       " ('USC00513117',\n",
       "  'KANEOHE 838.1, HI US',\n",
       "  21.4234,\n",
       "  -157.8015,\n",
       "  14.6,\n",
       "  0.9500000000000001),\n",
       " ('USC00516128',\n",
       "  'MANOA LYON ARBO 785.2, HI US',\n",
       "  21.3331,\n",
       "  -157.8025,\n",
       "  152.4,\n",
       "  0.89),\n",
       " ('USC00519281',\n",
       "  'WAIHEE 837.5, HI US',\n",
       "  21.45167,\n",
       "  -157.84888999999998,\n",
       "  32.9,\n",
       "  0.6399999999999999),\n",
       " ('USC00519523',\n",
       "  'WAIMANALO EXPERIMENTAL FARM, HI US',\n",
       "  21.33556,\n",
       "  -157.71139,\n",
       "  19.5,\n",
       "  0.52),\n",
       " ('USC00519397', 'WAIKIKI 717.2, HI US', 21.2716, -157.8168, 3.0, 0.29),\n",
       " ('USC00517948', 'PEARL CITY, HI US', 21.3934, -157.9751, 11.9, None)]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculate the total amount of rainfall per weather station for your trip dates \n",
    "# using the previous year's matching dates.\n",
    "\n",
    "# Sort this in descending order by precipitation amount and list the station, name, \n",
    "# latitude, longitude, and elevation\n",
    "\n",
    "\n",
    "start_date = '2016-02-14'\n",
    "end_date = '2016-02-29'\n",
    "\n",
    "\n",
    "stats_trip_data = session.query(Station.station, Station.name, Station.latitude, Station.longitude,\n",
    "                                Station.elevation, func.sum(Measurement.prcp)).\\\n",
    "                    filter(Measurement.station == Station.station).\\\n",
    "                    filter(Measurement.date >= start_date).filter(Measurement.date <= end_date).\\\n",
    "                    group_by(Station.station).order_by(func.sum(Measurement.prcp).desc()).all()\n",
    "stats_trip_data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(62.0, 69.15384615384616, 77.0)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a query that will calculate the daily normals \n",
    "# (i.e. the averages for tmin, tmax, and tavg for all historic data matching a specific month and day)\n",
    "\n",
    "def daily_normals(date):\n",
    "    \"\"\"Daily Normals.\n",
    "    \n",
    "    Args:\n",
    "        date (str): A date string in the format '%m-%d'\n",
    "        \n",
    "    Returns:\n",
    "        A list of tuples containing the daily normals, tmin, tavg, and tmax\n",
    "    \n",
    "    \"\"\"\n",
    "    \n",
    "    sel = [func.min(Measurement.tobs), func.avg(Measurement.tobs), func.max(Measurement.tobs)]\n",
    "    return session.query(*sel).filter(func.strftime(\"%m-%d\", Measurement.date) == date).all()\n",
    "    \n",
    "daily_normals(\"01-01\")[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(62.0, 70.89473684210526, 79.0),\n",
       " (56.0, 70.28813559322033, 79.0),\n",
       " (59.0, 70.48214285714286, 76.0),\n",
       " (62.0, 70.47272727272727, 77.0),\n",
       " (63.0, 70.79629629629629, 77.0),\n",
       " (60.0, 69.33962264150944, 77.0),\n",
       " (63.0, 70.14035087719299, 76.0),\n",
       " (63.0, 69.79629629629629, 76.0),\n",
       " (60.0, 70.15789473684211, 83.0),\n",
       " (62.0, 70.35593220338983, 81.0),\n",
       " (61.0, 68.56666666666666, 76.0),\n",
       " (61.0, 68.59649122807018, 76.0),\n",
       " (62.0, 69.89285714285714, 78.0),\n",
       " (58.0, 69.98148148148148, 77.0),\n",
       " (65.0, 70.65517241379311, 80.0),\n",
       " (67.0, 71.73333333333333, 79.0)]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# calculate the daily normals for your trip\n",
    "\n",
    "# push each tuple of calculations into a list called `normals`\n",
    "\n",
    "normals  = []\n",
    "\n",
    "\n",
    "\n",
    "# Set the start and end date of the trip\n",
    "\n",
    "start_date = '2016-02-14'\n",
    "end_date = '2016-02-29'\n",
    "\n",
    "\n",
    "\n",
    "# Use the start and end date to create a range of dates\n",
    "\n",
    "trip_day_list = pd.date_range(start_date,end_date)\n",
    "\n",
    "\n",
    "\n",
    "# Stip off the year and save a list of %m-%d strings\n",
    "\n",
    "date_time = trip_day_list.strftime('%m-%d')\n",
    "\n",
    "\n",
    "\n",
    "# Loop through the list of %m-%d strings and calculate the normals for each date\n",
    "\n",
    "for i in date_time:\n",
    "    normals.append(daily_normals(i)[0])\n",
    "    \n",
    "\n",
    "normals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['2016-02-14', '2016-02-15', '2016-02-16', '2016-02-17', '2016-02-18',\n",
       "       '2016-02-19', '2016-02-20', '2016-02-21', '2016-02-22', '2016-02-23',\n",
       "       '2016-02-24', '2016-02-25', '2016-02-26', '2016-02-27', '2016-02-28',\n",
       "       '2016-02-29'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trip_date = trip_day_list.strftime('%Y-%m-%d')\n",
    "trip_date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
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
       "      <th>Min Temp</th>\n",
       "      <th>Avg Temp</th>\n",
       "      <th>Max Temp</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Trip Dates</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2016-02-14</th>\n",
       "      <td>62.0</td>\n",
       "      <td>70.894737</td>\n",
       "      <td>79.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-02-15</th>\n",
       "      <td>56.0</td>\n",
       "      <td>70.288136</td>\n",
       "      <td>79.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-02-16</th>\n",
       "      <td>59.0</td>\n",
       "      <td>70.482143</td>\n",
       "      <td>76.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-02-17</th>\n",
       "      <td>62.0</td>\n",
       "      <td>70.472727</td>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2016-02-18</th>\n",
       "      <td>63.0</td>\n",
       "      <td>70.796296</td>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Min Temp   Avg Temp  Max Temp\n",
       "Trip Dates                               \n",
       "2016-02-14      62.0  70.894737      79.0\n",
       "2016-02-15      56.0  70.288136      79.0\n",
       "2016-02-16      59.0  70.482143      76.0\n",
       "2016-02-17      62.0  70.472727      77.0\n",
       "2016-02-18      63.0  70.796296      77.0"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the previous query results into a Pandas DataFrame and set 'Trip Dates' as index\n",
    "\n",
    "temp_df = pd.DataFrame(normals, columns=[\"Min Temp\", \"Avg Temp\", \"Max Temp\"], index=[i for i in range(16)])\n",
    "trip_df = pd.DataFrame(trip_date, columns=[\"Trip Dates\"], index=[i for i in range(16)])\n",
    "trip_data = pd.merge(trip_df, temp_df, left_index=True, right_index=True)\n",
    "trip_data.set_index('Trip Dates', inplace=True)\n",
    "trip_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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ERERyYRChYiIiGRCoUJEREQyoVAhIiIimVCoEBERkUwoVIiIiEgmFCpEREQkEwoVIiIikgmFChEREcmEQoWIiIhkQqFCREREMqFQISIiIplQqBAREZFMKFSIiIhIJhQqREREJBMKFSIiIpIJhQoRERHJhEKFiIiIZEKhQkRERDKhUCEiIiKZUKgQERGRTChUiIiISCYUKkRERCQTChUiIiKSCYUKERERyYRChYiIiGRCoUJEREQyEcz1hDiOefTRRzl+/Die5/HQQw/hnOORRx4B4Oabb+bBBx/E85RPRERE1rI5Q8XXv/51AH7nd36Hw4cP86u/+qs453jggQfYv38/jz32GE8++ST333//ohdWRERElq85mxfe/va388lPfhKAs2fP0tfXx9GjR9m3bx8A9913H88888zillJERESWvXn1WQRBwMMPP8yv/MqvcODAAZxzGGMAKJVKjI+PL2ohRUREZPmbs/uj4eGHH2ZwcJAPfvCD1Gq15v3lcpnu7u55vcbAwMDCSzgPYRgtyusuZ6rz6rfW6guq81qx1urcjvp6RIyOjGa+3e3v75/18TlDxZe+9CUuXLjABz7wATo6OjDGcNttt3H48GH279/PoUOHuPPOOzMpzLV49jDkcvPORqtCGEaq8yq31uoLqvNasdbq3Lb6Wujp7VmU7e5s5qzp/fffzyOPPMKP/diPEUURH//4x9m9ezePPvooYRiyZ88eDhw4sBRlFRERkWVszlBRLBZ57LHHrrj/c5/73KIUSERERFYmTS4hIiIimVCoEBERkUwoVIiIiEgmFCpEREQkEwoVIiIikgmFChEREcmEQoWIiIhkQqFCREREMqFQISIiIplQqBAREZFMKFSIiIhIJhQqREREJBMKFSIiIpIJhQoRERHJhEKFiIiIZEKhQkRERDKhUCEiIiKZUKgQERGRTChUiIiISCYUKkRERCQTChUiIiKSCYUKERERyYRChYiIiGRCoUJEREQyoVAhIiIimVCoEBERkUwoVIiIiEgmFCpEREQkEwoVIiIikolgtgejKOLTn/40Z86cIQxDPvjBD7Jnzx4eeeQRAG6++WYefPBBPE/ZREREZK2bNVR8+ctfpre3l0996lMMDw/zIz/yI+zdu5cHHniA/fv389hjj/Hkk09y//33L1V5RUREZJmatYnhu77ru/jxH//x5m3f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d9ftawz1zdOmTcPf35/c3FxefvllevTo4ehh3Xcqbb9U23aBPrZfyu2paN++PefPn2fatGlA/S3G9eyvf/0rSUlJTJo0ieLiYj744AP27duHwWDQ5W92ixYtIiwsjLS0NLy8vCgvL6e2thbQ72+yy5cvJywsjEmTJpGcnMwrr7zC3r17dfsaAyxZsoTo6GjmzJnDuHHjOHz4sKOH1CxU2n6ptu0CfWy/9PuOvIPIyEgWLVpEXFwczz33HGlpadTU1GCxWBw9tPvCz88Pi8XCq6++ioeHB25ubsyfP9/2l47eBAcHExYWhtlspqCggPfff5+xY8eye/duXfZCfXN0dDStWrXi8ccfZ8aMGbz11lvk5eXpttnX15fExETbx8ePH7/l686yAW4slbZfZrNZqW0XQJs2bZx++6XcpOLSpUtcvHiRgQMHUlFRQWVlJW5ubrrdhejn50dJSQn+/v6MGjWKp556iilTpthus64nVquVqKgoSktLGTlyJJqm8eqrrzJmzBi2b9+uqw1vw7Hma9euYTabyc/P59KlS2iaRq9evRgxYgSFhYWOHqZdNTRXVlYSERFB+/btAQgICLAtcPjFF19w/Phxp9kA343VaqWgoMC21pLet183v68DAgK4ePGi7rddN7+vo6OjuXz5slNvv/TxTvwZGRkZJCQkkJKSAtTP9P/5z3+SmZnJwoUL2bZtG9OmTWPp0qW62RDd3Ny7d2/279/P/v37ycnJ4aGHHqKmpobq6mrdHJvMyMggPj6e4cOH8/jjj9OrVy+qqqqIj4/HZDKhaRru7u66aIX6464zZ87EarXi6upKamoqR48eZePGjTzzzDOEh4dTVVVFcXGxo4dqN7c3p6WlERoaCkBNTQ0xMTF88803/O1vf2PevHkOHq19NDRD/Yl706ZNIzY2lmXLluly+3Vzr9FoZPz48Rw+fJh9+/bRs2dPXW67bn5fm0wm0tLS6Nu3L5WVlSQmJjrl9stl5syZCxw9iPtp165dvP/++wQFBREbG4uXlxfbt29nxIgRJCUl8eijj/Lggw/i5eXl6KHaTUOzv78/nTt3JikpidOnT3PkyBE+++wzDh8+zLhx4/D393f0UO1i165dbN68mcDAQOLi4jCZTOzbt49jx45RUFDAzp07GT9+PAEBAY4eql288847+Pv7k56eTk5ODufOnePFF18k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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the daily normals as an area plot with `stacked=False`\n",
    "\n",
    "trip_data.plot(kind='area', stacked=False, rot=45)\n",
    "# tick_values = trip_date.index.strftime('%Y-%m-%d')\n",
    "# plt.xticks(ticks=trip_date.index, labels=tick_values)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernel_info": {
   "name": "python3"
  },
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
   "version": "3.7.6"
  },
  "nteract": {
   "version": "0.12.3"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
