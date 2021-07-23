const logger = require('../logger');

class ErrorResponse extends Error {
	constructor(message, statusCode) {
		super(message);
		this.statusCode = statusCode;
	}
}

const errorHandler = (err, req, res, next) => {
	let error = { ...err };

	error.message = err.message;

	// Log to console for dev + Logz.io
	logger.error({ message: err, division: 'errorHandler', stack: err.stack });

	// Mongoose casting error
	if (err.name === 'CastError') {
		const message = `Resource not found with ${err.value}`;
		error = new ErrorResponse(message, 404);
	}

	// Mongoose duplicate key
	if (err.code === 11000) {
		const message = 'Duplicate field value entered';
		error = new ErrorResponse(message, 400);
	}

	// Mongoose validation error
	if (err.name === 'ValidationError') {
		let message = 'Missing body params: ';
		message += Object.values(err.errors).map((val) => val.path);
		error = new ErrorResponse(message, 400);
	}

	// Custom throw message error, used for nested functions using Mongoose to find documents
	if (!err.message) {
		const message = err;
		error = new ErrorResponse(message, 404);
	}

	res.status(error.statusCode || 500).json({
		success: false,
		error: 'An Error Has Occurred, Please Try Again Later',
	});
};

module.exports = {
	errorHandler: errorHandler,
	ErrorResponse: ErrorResponse,
};
