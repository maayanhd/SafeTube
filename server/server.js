const express = require('express');
const app = express();
const cors = require('cors');
const youtube = require('./components/youtube/youtubeRoutes');
const morgan = require('morgan');
const { Database } = require('./components/database-v2');

const { errorHandler } = require('./components/backend-error-handler');

(async () => {
	const { NODE_ENV, DB_STREAMERS_USER, DB_STREAMERS_PASSWORD, DB_STREAMERS_URL } = process.env;
	const mongoServerUrl = `mongodb+srv://${DB_STREAMERS_USER}:${DB_STREAMERS_PASSWORD}@${DB_STREAMERS_URL}`;
	await Database.init(mongoServerUrl);

	app.use(morgan(NODE_ENV === 'production' ? 'short' : 'dev'));
	app.use(express.json());
	app.use(errorHandler);

	app.use('/api/v1/liveUpdater/youtube', youtube);
})();

module.exports = app;
