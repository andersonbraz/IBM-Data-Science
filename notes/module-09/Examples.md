# Simple Example - API Foursquare 


```python
import json, requests   ## interactions of requests and json
import pprint           ## pretty json

url = 'https://api.foursquare.com/v2/venues/explore'

params = dict(
client_id = 'YOUR CLIENT ID HERE',
client_secret = 'YOUR CLIENT SECRET HERE',
v = '20180323',
ll = '40.7243,-74.0018',
query = 'coffee',
limit = 1
)
resp = requests.get(url=url, params=params)
data = json.loads(resp.text)

pprint.pprint(data)
```

    {'meta': {'code': 200, 'requestId': '5fee1b8878a7474f015224cb'},
     'response': {'groups': [{'items': [{'reasons': {'count': 0,
                                                     'items': [{'reasonName': 'globalInteractionReason',
                                                                'summary': 'This '
                                                                           'spot '
                                                                           'is '
                                                                           'popular',
                                                                'type': 'general'}]},
                                         'referralId': 'e-0-573498df498e6df2eb8b36a7-0',
                                         'venue': {'beenHere': {'count': 0,
                                                                'lastCheckinExpiredAt': 0,
                                                                'marked': False,
                                                                'unconfirmedCount': 0},
                                                   'categories': [{'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
                                                                            'suffix': '.png'},
                                                                   'id': '4bf58dd8d48988d1e0931735',
                                                                   'name': 'Coffee '
                                                                           'Shop',
                                                                   'pluralName': 'Coffee '
                                                                                 'Shops',
                                                                   'primary': True,
                                                                   'shortName': 'Coffee '
                                                                                'Shop'}],
                                                   'contact': {},
                                                   'hereNow': {'count': 0,
                                                               'groups': [],
                                                               'summary': 'Nobody '
                                                                          'here'},
                                                   'id': '573498df498e6df2eb8b36a7',
                                                   'location': {'address': '154 '
                                                                           'Prince '
                                                                           'St',
                                                                'cc': 'US',
                                                                'city': 'New York',
                                                                'country': 'United '
                                                                           'States',
                                                                'crossStreet': 'btwn '
                                                                               'W '
                                                                               'Broadway '
                                                                               '& '
                                                                               'Thompson',
                                                                'distance': 176,
                                                                'formattedAddress': ['154 '
                                                                                     'Prince '
                                                                                     'St '
                                                                                     '(btwn '
                                                                                     'W '
                                                                                     'Broadway '
                                                                                     '& '
                                                                                     'Thompson)',
                                                                                     'New '
                                                                                     'York, '
                                                                                     'NY '
                                                                                     '10012',
                                                                                     'United '
                                                                                     'States'],
                                                                'labeledLatLngs': [{'label': 'display',
                                                                                    'lat': 40.72581964106336,
                                                                                    'lng': -74.00119185447693},
                                                                                   {'label': 'entrance',
                                                                                    'lat': 40.72582,
                                                                                    'lng': -74.001153}],
                                                                'lat': 40.72581964106336,
                                                                'lng': -74.00119185447693,
                                                                'postalCode': '10012',
                                                                'state': 'NY'},
                                                   'name': 'La Colombe '
                                                           'Torrefaction',
                                                   'photos': {'count': 0,
                                                              'groups': []},
                                                   'stats': {'checkinsCount': 0,
                                                             'tipCount': 0,
                                                             'usersCount': 0,
                                                             'visitsCount': 0},
                                                   'verified': False}}],
                              'name': 'recommended',
                              'type': 'Recommended Places'}],
                  'headerFullLocation': 'SoHo, New York',
                  'headerLocation': 'SoHo',
                  'headerLocationGranularity': 'neighborhood',
                  'query': 'coffee',
                  'suggestedBounds': {'ne': {'lat': 40.727169470956085,
                                             'lng': -74.00255064150711},
                                      'sw': {'lat': 40.724469811170636,
                                             'lng': -73.99983306744674}},
                  'suggestedFilters': {'filters': [{'key': 'openNow',
                                                    'name': 'Open now'},
                                                   {'key': 'price',
                                                    'name': '$-$$$$'}],
                                       'header': 'Tap to show:'},
                  'suggestedRadius': 600,
                  'totalResults': 63,
                  'warning': {'text': "There aren't a lot of results for "
                                      '"coffee." Try something more general, reset '
                                      'your filters, or expand the search area.'}}}

