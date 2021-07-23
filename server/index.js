const app = require('./server.js');
const server = require('http').Server(app);
const logger = require('./components/logger');

const port = process.env.PORT || 8082;

app.listen(port, () => {
	logger.info({
		message: `Server running in ${process.env.NODE_ENV} mode on port ${port}.`,
		division: 'SafeTube-Backend',
	});
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (error, promise) => {
	logger.error({ message: error, division: 'SafeTube-Backend' });

	server.close(() => {
		process.exit(0);
	});

	// If server hasn't finished in 1000ms, shut down process
	setTimeout(() => {
		process.exit(0);
	}, 1000).unref(); // Prevents the timeout from registering on event loop
});
