const { createLogger, format, transports } = require('winston');
const { combine, timestamp, printf } = format;


const errorFormat = printf((info) => {
	let result = '';
	if (info.stack) {
		result += `${info.stack}`;
	} else {
		result = `
    ${info.timestamp}
    Level: ${info.level.toUpperCase()},
    Division: ${info.division || 'Not specified'},
    Message: ${info.message},
    Env: ${process.env.NODE_ENV || info.env}
    `;
	}

	return result;
});

const logger = createLogger({
	format: combine(format.simple(), timestamp(), errorFormat, format.colorize({ all: true })),
	// transports: [new transports.Console({ level: 'silly' }), logzioWinston],
	transports: [new transports.Console({ level: 'silly' })],
});

module.exports = logger;
