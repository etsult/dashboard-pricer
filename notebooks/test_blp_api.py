import blpapi
from blpapi import SessionOptions, Session

SESSION_HOST = "localhost"
SESSION_PORT = 8194

options = SessionOptions()
options.setServerHost(SESSION_HOST)
options.setServerPort(SESSION_PORT)

session = Session(options)
session.start()
session.openService("//blp/refdata")

service = session.getService("//blp/refdata")
request = service.createRequest("ReferenceDataRequest")

request.getElement("securities").appendValue("AAPL US Equity")
request.getElement("fields").appendValue("PX_LAST")

session.sendRequest(request)

while True:
    ev = session.nextEvent(500)
    for msg in ev:
        print(msg)
    if ev.eventType() == blpapi.Event.RESPONSE:
        break

session.stop()
