#!/usr/bin/env python

import requests
import json
import time,sys

from requests.auth import HTTPBasicAuth
from collections import OrderedDict
from urllib import urlencode

import matplotlib.pyplot as plt
import numpy as np
import optparse
import random 
import getpass
#import initExample ## Add path to library (just for examples; you do not need this)
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
#import tflearn
from numpy import linalg as la 

parser = optparse.OptionParser("usage: %prog -u <username> [-p <password> -r <minfreq,maxfreq> -t <timeresol> -f <frequency_resol>]")
parser.add_option("-u", "--user", dest="username",
                  type="string",
                  help="API username")
parser.add_option("-p", "--pass", dest="password",
                    type="string", help="API password")

parser.add_option("-r", "--range", dest="frange",
                    type="string", help="frequency range separated by commas")

parser.add_option("-t", "--tresol", dest="tresol",
                    type="string", help="time resolution")

parser.add_option("-f", "--fresol", dest="fresol",
                    type="string", help="frequency resolution")

(options, args) = parser.parse_args()
if not options.username:
   parser.error("Username not specified")

if not options.password:
   options.password = getpass.getpass('Password:')

# Electrosense API Credentials 
username=options.username
password=options.password

# Electrosense API
MAIN_URI ='https://test.electrosense.org/api'
SENSOR_LIST = MAIN_URI + '/sensor/list/'
SENSOR_AGGREGATED = MAIN_URI + "/spectrum/aggregated"

r = requests.get(SENSOR_LIST, auth=HTTPBasicAuth(username, password))

if r.status_code != 200:
    print r.content
    exit(-1)

slist_json = json.loads(r.content)

senlist={}
status=[" (off)", " (on)"]

for i, sensor in enumerate(slist_json):
    print "[%d] %s (%d) - Sensing: %s" % (i, sensor['name'], sensor['serial'], sensor['sensing'])
    senlist[sensor['name']+status[int(sensor['sensing'])]]=i

print ""
pos = int( raw_input("Please enter the sensor: "))

print ""
print "   %s (%d) - %s, type:%s, frontend:%s, firmware:%s" % (slist_json[pos]['name'], slist_json[pos]['serial'], slist_json[pos]['sensing'], slist_json[pos]['type'], slist_json[pos]['frontend'], slist_json[pos]['firmware'])


# Ask for 5 minutes of aggregatd spectrum data

def get_spectrum_data (sensor_id, timeBegin, timeEnd, aggFreq, aggTime, minfreq, maxfreq):
    
    params = OrderedDict([('sensor', sensor_id),
                          ('timeBegin', timeBegin),
                          ('timeEnd', timeEnd),
                          ('freqMin', int(minfreq)),
                          ('freqMax', int(maxfreq)),
                          ('aggFreq', aggFreq),
                          ('aggTime', aggTime),
                          ('aggFun','AVG')])


    r = requests.get(SENSOR_AGGREGATED, auth=HTTPBasicAuth(username, password), params=urlencode(params))

    
    if r.status_code == 200:
        return json.loads(r.content)
    else:
        print "Response: %d" % (r.status_code)
        return None

sp1 = None
sp2 = None
sp3 = None    

epoch_time = int(time.time())
timeBegin = epoch_time - (3600*24*2)
#timeEnd = timeBegin + (3600*20*2)
timeEnd = timeBegin + (60*4)
if not options.fresol:
    freqresol = int(100e3)
else:
    freqresol = int(float(options.fresol))

if not options.tresol:
    tresol = int(60)
else:
    tresol = int(float(options.tresol))

if not options.frange:
    minfreq = 50e6 
    maxfreq = 1500e6
else:
    minfreq = int(float(options.frange.split(",")[0])) 
    maxfreq = int(float(options.frange.split(",")[1])) 

senid = slist_json[pos]['serial'] 
response = get_spectrum_data (slist_json[pos]['serial'], timeBegin, timeEnd, freqresol, tresol, minfreq, maxfreq)
data=np.array(response['values'])
print "Data:",data.shape

print data[0]

