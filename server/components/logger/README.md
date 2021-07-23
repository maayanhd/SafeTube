# boilerplate-logger

here will be the logger that all the micro services will use to output their logs to winston and logzio.
There are 2 ways to use the logger after installing the npm package:

1. logger.debug({ message: 'Hey! Log something?',division: "Notification MicroService"});
   explanation: logger.debug => debug is the level of the log, all the levels written at the end.
                message => added info to the bug (example: Error occurred in the HTTP request).
                division => where the bug happened(example: Notification MicroService).

2. logger.error(new Error("message")) / try{}catch{err => logger.error(err)}, This format record stack trace.  
   explanation: logger.error => error is the level of the log, all the levels written at the end.
                message => added info to the bug (example: Error occurred in the HTTP request).

log levels from high to low:
error: 0,
warn: 1,
info: 2,
http: 3,
verbose: 4,
debug: 5,
silly: 6
