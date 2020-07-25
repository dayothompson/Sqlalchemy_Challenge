import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

from flask import Flask, jsonify


# Setup Database
engine = create_engine("sqlite:///Resources/hawaii.sqlite")

# Reflect an existing database into a new model
Base = automap_base()

# Reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
Measurement = Base.classes.measurement
Station = Base.classes.station

# Create our session (link) from Python to the DB
session = Session(engine)


# Flask Setup
app = Flask(__name__)



#################################################
# Flask Routes
#################################################

@app.route("/")
def welcome():
    """List all available api routes."""
    return (
        f"/api/v1.0/precipitation<br/>"
        f"/api/v1.0/stations<br/>"
        f"/api/v1.0/tobs<br/>"
        f"/api/v1.0/start<br/>"
        f"/api/v1.0/start/end<br/>"
    )


@app.route("/api/v1.0/precipitation")
def precipitation():

    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Indicate the time period
    start_date = "2016-08-24"
    end_date = "2017-08-23"


    # Query the date and preciptation columns of the Measurement table for the past year
    prep_data = session.query(Measurement.date, Measurement.prcp).\
                            filter(Measurement.date >= start_date).\
                            filter(Measurement.date <= end_date).all()

    session.close()

    # Convert the query to JSON format
    prep_list = []

    for date, prep in prep_data:
        prep_dict = {}
        prep_dict[date] = prep
        prep_list.append(prep_dict)


    return jsonify(prep_list)


@app.route("/api/v1.0/stations")
def stations():

    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Query the station, name, latitude, longitude, elevation columns of the Station table
    stats_data = session.query(Station.station, Station.name, Station.latitude, Station.longitude,
                                Station.elevation).all()

    session.close()

    # Convert the query to JSON format
    stats_list = []

    for station, name, latitude, longitude, elevation in stats_data:
        stats_dict = {}
        stats_dict['station'] = station
        stats_dict['name'] = name
        stats_dict['latitude'] = latitude
        stats_dict['longitude'] = longitude
        stats_dict['elevation'] = elevation
        stats_list.append(stats_dict)

    return jsonify(stats_list)


@app.route("/api/v1.0/tobs")
def tobs():

    stats_hi = "USC00519281"
    start_date = "2016-08-24"
    end_date = "2017-08-23"

    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Query the date and temperature columns of the Measurement table for the past year
    temp_stats_data = session.query(Measurement.date, Measurement.tobs).\
                        filter(Measurement.station == "USC00519281").\
                        filter(Measurement.date >= start_date).\
                        filter(Measurement.date <= end_date).all()


    session.close()


    # Convert the query to JSON format
    temp_stats_list = []
    # temp_stats_list = list(results)
    for date, temp in temp_stats_data:
        temp_stats_dict = {}
        temp_stats_dict[date] = temp
        temp_stats_list.append(temp_stats_dict)


    return jsonify(temp_stats_list)



@app.route("/api/v1.0/<start>", defaults={"end": "2017-08-23"})
@app.route("/api/v1.0/<start>/<end>")
def temp(start, end):


    # Create our session (link) from Python to the DB
    session = Session(engine)

    # Query the minimum, average and maximum temperature for the given dates
    temp_data = session.query(func.min(Measurement.tobs),\
                              func.avg(Measurement.tobs), func.max(Measurement.tobs)).\
                                filter(Measurement.date >= start).\
                                filter(Measurement.date <= end).all()

    session.close()

    # Convert the query to JSON format
    temp_data_list = []

    for tmin, tavg, tmax in temp_data:
        temp_data_dict = {}
        temp_data_dict["TMIN"] = tmin
        temp_data_dict["TAVG"] = tavg
        temp_data_dict["TMAX"] = tmax
        temp_data_list.append(temp_data_dict)


        return jsonify(temp_data_list)



if __name__ == '__main__':
    app.run(debug=True, port=5001)