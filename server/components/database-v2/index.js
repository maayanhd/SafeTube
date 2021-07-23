const mongoose = require('mongoose');
const logger = require('../logger');
const streamerPlatformEnum = ['twitch', 'mixer', 'youtube', 'facebook', 'dlive'];
const importMethods = ['import', 'manual'];
const liveStatuses = ['offline', 'online', 'uncertain'];
const userPlatformEnum = ['twitch', 'discord', 'google-oauth2', 'facebook', 'auth0', 'Username-Password-Authentication'];
const daysEnum = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'saturday'];

// ! STREAMERS SCHEMA - START
const socialSchema = new mongoose.Schema(
	{
		instagram: {
			type: String, //TODO: figure out Regexp validator for http://www.instagram.com/
			default: null,
			trim: true,
		},
		twitter: {
			type: String, //TODO: figure out Regexp validator for http://www.twitter.com/
			default: null,
			trim: true,
		},
	},
	{ _id: false }
);

const fortniteRawDataSchema = new mongoose.Schema(
	{
		interface: {
			deathKills: {
				type: Boolean,
				default: false,
			},
			earlyAccess: {
				type: Boolean,
				default: false,
			},
			deathHash: {
				type: Boolean,
				default: false,
			},
			kills: {
				type: Boolean,
				default: false,
			},
			remaining: {
				type: Boolean,
				default: false,
			},
			respawn: {
				type: Boolean,
				default: false,
			},
			spectateKills: {
				type: Boolean,
				default: false,
			},
			timer: {
				type: Boolean,
				default: false,
			},
			hasVictory: {
				type: Boolean,
				default: false,
			},
			hasViewIcon: {
				type: Boolean,
				default: false,
			},
			isTeamGame: {
				type: Boolean,
				default: false,
			},
			isCompetitive: {
				type: Boolean,
				default: false,
			},
		},
		values: {
			kills: {
				type: Number,
				default: null,
			},
			remaining: {
				type: Number,
				default: null,
			},
			timer: {
				type: Number,
				default: null,
			},
			competitiveRank: {
				type: Number,
				default: null,
			},
		},
	},
	{ timestamps: {} }
);
//TODO dont know if need to remove this schema, no use in current code.
const fortniteFrameSchema = new mongoose.Schema(
	{
		inGame: {
			// BASE
			type: Boolean,
			default: false,
		},
		gameResult: {
			// BASE
			type: Number,
			default: 0,
			enum: [0, 1, 2, 3],
		},
		firstGame: {
			type: Boolean,
			default: true,
		},
		squadSize: {
			// BASE
			type: Number,
			default: 1,
			min: 1,
		},
		eliminations: {
			type: Number,
			default: 0,
			min: 0,
		},
		timer: {
			type: Number,
			default: 0,
			min: 0,
		},
		remainingPlayers: {
			type: Number,
			default: 100,
			max: 100,
			min: 1,
		},
		statistics: {
			// BASE
			type: mongoose.Schema.Types.Mixed,
			default: {},
		},
		traits: {
			type: mongoose.Schema.Types.Mixed,
			default: {},
		},
	},
	{ _id: false, timestamps: true }
);

const streamerTraitsSchema = new mongoose.Schema(
	{
		sex: {
			type: Number,
			default: 0,
			enum: [0, 1, 2, 3],
			get: (v) => Math.round(v),
			set: (v) => Math.round(v),
		},
		interactive: {
			value: {
				type: Boolean,
				default: false,
			},
			confidence: {
				type: Number,
				default: 0,
			},
		},
	},
	{ _id: false }
);

const streamerObj = {
	name: {
		type: String,
		required: true,
	},
	agentId: {
		type: String,
		default: null,
	},
	viewers: {
		type: Number,
		default: 0,
		min: 0,
		get: (v) => Math.round(v),
		set: (v) => Math.round(v),
	},
	timeout: {
		type: Date,
		default: new Date(),
	},
	currentGame: {
		type: String,
		default: 'unsupported game',
	},
	inGame: {
		type: Boolean,
		default: false,
	},
	games: {
		type: Object,
		default: {},
	},
	rank: {
		type: Number,
		default: 0,
	},
	logo: {
		type: String,
		default: null,
	},
	isBlocked: {
		type: Boolean,
		default: false,
	},
	liveStatus: {
		type: String,
		default: 'offline',
		enum: liveStatuses,
	},
	followers: {
		type: Number,
		default: 0,
		min: 0,
		get: (v) => Math.round(v),
		set: (v) => Math.round(v),
	},
	previewThumbnail: {
		type: String,
		default: null,
	},
	updateIteration: {
		type: Number,
		default: 0,
		min: 0,
		get: (v) => Math.round(v),
		set: (v) => Math.round(v),
	},
	isTracked: {
		type: Boolean,
		default: false,
	},
	platform: {
		name: {
			type: String,
			required: true,
			enum: streamerPlatformEnum,
		},
		id: {
			type: mongoose.Schema.Types.Mixed,
			required: true,
		},
	},
	social: { type: socialSchema, default: () => ({}) },
	streamVideoUrl: {
		type: String,
		default: null,
	},
	traits: { type: streamerTraitsSchema, default: () => ({}) }, // NOTE: includes => sex,isFunny,
	metadata: {
		chatSpeedSamplingInterval: {
			type: Number,
			default: 60,
			min: 1,
		},
		avgChatSpeed: {
			// per 1min/10sec/other
			type: mongoose.SchemaTypes.Decimal128,
			default: 0,
		},
		title: {
			type: String,
			default: null,
		},
		description: {
			type: String,
			default: null,
		},
		chatId: {
			type: String,
			default: null,
		},
	},
	statistics: {
		averageTopViewers: {
			type: Number,
			default: 0,
			min: 0,
		},
		sessionTopViewersArray: {
			type: [Number],
			default: [],
			validate: {
				validator: function (array) {
					return array.length <= 5;
				},
				message: (props) => `${props.value} exceeds maximum array size (5)!`,
			},
		},
		averageViewers: {
			type: Number,
			default: 0,
		},
		averageViewersCounter: {
			type: Number,
			default: 0,
			get: (v) => Math.round(v),
			set: (v) => Math.round(v),
		},
	},
};

const streamerObjectOptions = { timestamps: true };

const platformSchema = new mongoose.Schema(
	{
		platform: {
			type: String,
			default: null,
			enum: userPlatformEnum,
		},
		pid: {
			type: String,
			default: null,
		},
		name: {
			type: String,
			default: null,
		},
		token: {
			type: String,
			default: null,
		},
		refreshToken: {
			type: String,
			default: null,
		},
		avatar: {
			type: String,
			default: null,
		},
		tokenExpirationDate: {
			type: Date,
			default: null,
		},
	},
	{ _id: false, timestamps: {} }
);

const favoritesSchema = new mongoose.Schema(
	{
		name: {
			type: String,
			required: true,
		},
		platform: {
			name: {
				type: String,
				required: true,
				enum: streamerPlatformEnum,
			},
			id: {
				type: mongoose.Schema.Types.Mixed,
				required: true,
			},
		},
		method: {
			type: String,
			required: true,
			enum: importMethods,
		},
		isFollowing: {
			type: Boolean,
			default: true,
		},
		isSubscribed: {
			type: Boolean,
			default: false,
		},
	},
	{ _id: false }
);

const userEventsSchema = new mongoose.Schema(
	{
		name: {
			type: String,
			required: true,
		},
		startTime: {
			type: Date,
			default: new Date(),
		},
		endTime: {
			type: Date,
			default: new Date(),
		},
		metadata: {
			type: mongoose.SchemaTypes.Mixed,
			default: null,
		},
	},
	{ _id: false }
);

const usersSchema = new mongoose.Schema(
	{
		name: {
			type: String,
			default: null,
		},
		email: {
			type: String,
			default: null,
			match: [/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/, 'Please use a valid email address'],
		},
		guest: {
			type: Boolean,
			default: false,
		},
		ip: {
			type: String,
			default: null,
		},
		locale: {
			type: String,
			default: null,
		},
		userEvents: [userEventsSchema],
		favorites: [favoritesSchema],
		connectedPlatforms: [platformSchema],
	},
	{ timestamps: {} }
);

// ! USERS SCHEMA - END

// ! AGENTS DATA SCHEMA - START

const fortniteDataSchema = new mongoose.Schema(
	{
		data: { type: fortniteRawDataSchema, default: () => ({}) },
		agentId: {
			type: String,
			required: true,
			trim: true,
		},
		streamer: {
			name: {
				type: String,
				required: true,
				trim: true,
			},
			platform: {
				type: String,
				required: true,
				trim: true,
			},
		},
	},
	{ timestamps: {} }
);

// ! STATISTICS SCHEMA - START
const statisticsDashboardSchema = new mongoose.Schema(
	{
		microservice: {
			type: String,
			required: true,
		},
		ttl: {
			type: Number,
			require: true,
			default: 0,
		},
		endpoint: {
			type: String,
			required: true,
		},
	},
	{ timestamps: {} }
);
// ! STATISTICS SCHEMA - END

// ! TOKEN SCHEMA - START
const tokenSchema = new mongoose.Schema(
	{
		used: {
			type: Boolean,
			default: false,
		},
	},
	{ timestamps: {} }
);
// ! TOKEN SCHEMA - END
const scheduleDayObject = {
	avgStart: {
		type: String,
		default: '00:00',
	},
	avgDuration: {
		type: Number,
		default: 1,
		min: 1,
	},
	avgDurationCount: {
		type: Number,
		default: 1,
		min: 1,
	},
};

const scheduleSchema = new mongoose.Schema(
	{
		streamerDocId: {
			type: mongoose.Types.ObjectId,
		},
		schedule: {
			sunday: scheduleDayObject,
			monday: scheduleDayObject,
			tuesday: scheduleDayObject,
			wednesday: scheduleDayObject,
			thursday: scheduleDayObject,
			saturday: scheduleDayObject,
		},
		currentStream: {
			startTime: {
				type: String,
				default: '00:00',
			},
			startDay: {
				type: Date,
			},
			Duration: {
				type: Number,
				default: 0,
				min: 0,
			},
		},
		isUpdatedToday: {
			type: Boolean,
			default: false,
		},
	},
	{ timestamps: {} }
);

const messageSchema = new mongoose.Schema(
	{
		// senderName: {
		// 	type: String,
		// 	required: true,
		// },
		text: {
			type: String,
			required: true,
		},
		tags: {
			type: mongoose.Schema.Types.Mixed,
			default: {},
		},
	},
	{ _id: false }
);

const chatFrameSchema = new mongoose.Schema(
	{
		name: {
			type: String,
			required: true,
		},
		platform: {
			type: String,
			required: true,
			enum: streamerPlatformEnum,
		},
		// game: {
		// 	type: String,
		// 	default: null,
		// 	enum: gamesEnum,
		// },
		// range: {
		// 	start: {
		// 		type: Date,
		// 		default: new Date(),
		// 	},
		// 	end: {
		// 		type: Date,
		// 		default: new Date(),
		// 	},
		// },
		// messageCount: {
		// 	type: Number,
		// 	default: 0,
		// 	min: 0,
		// },
		date: {
			type: Date,
			default: new Date(),
		},
		rawData: { type: messageSchema, default: () => ({}) },
	},
	{ timestamps: {} }
);

const gamePlatformsSchema = new mongoose.Schema(
	{
		twitch: {
			id: {
				type: Number,
				default: null,
				get: (v) => Math.round(v),
				set: (v) => Math.round(v),
			},
			name: {
				type: String,
				default: null,
			},
		},
		youtube: {
			id: {
				type: String,
				default: null,
			},
			name: {
				type: String,
				default: null,
			},
		},
		facebook: {
			id: {
				type: Number,
				default: null,
			},
			name: {
				type: String,
				default: null,
			},
		},
		dlive: {
			id: {
				type: Number,
				default: null,
				get: (v) => Math.round(v),
				set: (v) => Math.round(v),
			},
			name: {
				type: String,
				default: null,
			},
		},
	},
	{ _id: false }
);

const gameSchema = new mongoose.Schema(
	{
		name: {
			type: String,
			required: true,
		},
		imageUrl: {
			type: String,
			default: null,
		},
		isSupported: {
			type: Boolean,
			default: false,
		},
		platforms: { type: gamePlatformsSchema, default: () => ({}) },
	},
	{ timestamps: {} }
);

const iterationSchema = new mongoose.Schema(
	{
		platform: {
			type: String,
			required: true,
		},
		currentIteration: {
			type: Number,
			default: 0,
			min: 0,
			get: (v) => Math.round(v),
			set: (v) => Math.round(v),
		},
		interval: {
			type: Number,
			default: 3600,
			get: (v) => Math.round(v),
			set: (v) => Math.round(v),
		},
	},
	{ timestamps: {} }
);

const likeSchema = new mongoose.Schema(
	{
		userDocId : {
			type: String,
			required: true,
		},
		streamerPlatform: {
			name: {
				type: String,
				required: true,
				enum: streamerPlatformEnum,
			},
			id: {
				type: mongoose.Schema.Types.Mixed,
				required: true,
			},
		}, 
		isCurrentSession: {
			type: Boolean,
			default: true,
		},
	},
	{ timestamps: {}, _id: false  }
);

/**
 * Returns the options that are to be used when connecting with mongoose to the database
 * @param {String} db The name of the database
 */
const getConnectionOptions = (db) => {
	return {
		dbName: db,
		poolSize: 100,
		useNewUrlParser: true,
		useUnifiedTopology: true,
		useFindAndModify: false,
		useCreateIndex: true,
	};
};

let connection = null;

/**
 * Initializes the mongoose connections pool
 * @param {String} uri Mongodb connection url
 * @param {String} db Name of the DB we wish to connect with
 */
const init = async (uri = null, db = 'YoutubeSafetyDetector') => {
	if (!uri) {
		throw new Error('No mongodb url sent');
	}

	if (connection) {
		connection.close();
	}

	connection = await mongoose.createConnection(uri, getConnectionOptions(db)).catch((error) => {
		throw error;
	});

	logger.info({
		message: `MongoDB Connected: ${connection.host}`,
		division: 'database package',
		env: process.env.NODE_ENV,
	});
};

/**
 * Streamers model connection
 */
const streamers = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	if (!connection.models.Streamer) {
		const gamesModel = await games();
		const allSupportedGames = await gamesModel.find({ isSupported: true });

		const streamerSchema = new mongoose.Schema(streamerObj, streamerObjectOptions);
		addStreamerSchemaHooks(streamerSchema, allSupportedGames);
		const streamersModel = connection.model('Streamer', streamerSchema);
		return streamersModel;
	} else {
		return connection.models.Streamer;
	}

	//* Old Code
	// let streamersModel, streamerSchema;

	// if (game) {
	// 	if (!connection.models[`Streamer-${game}`]) {
	// 		// make sure to not override the existing model
	// 		// insert or update a streamer according to a specific game using the streamer JSON object combined with the game's schema
	// 		streamerSchema = new mongoose.Schema(Object.assign(streamerObj, { games: { [game]: { type: getGameSchema(game), default: () => ({}) } } }), streamerObjectOptions);
	// 		addStreamerSchemaHooks(streamerSchema);
	// 		streamersModel = connection.model(`Streamer-${game}`, streamerSchema, 'streamers');
	// 	} else {
	// 		streamersModel = connection.models[`Streamer-${game}`];
	// 	}
	// } else {
	// if (!connection.models.Streamer) {
	// // make sure to not override the existing model
	// //used for pulling the streamer with the old (deprecated) schema
	// let games = {};
	// for (const game of gamesEnum) {
	// 	Object.assign(games, { [game]: { type: getGameSchema(game), default: () => ({}) } });
	// }

	// streamerSchema = new mongoose.Schema(Object.assign(streamerObj, { games: games }), streamerObjectOptions);
	// streamerSchema = new mongoose.Schema(streamerObj, streamerObjectOptions);
	// addStreamerSchemaHooks(streamerSchema);
	// streamersModel = connection.model('Streamer', streamerSchema);
	// } else {
	// 	streamersModel = connection.models.Streamer;
	// }
	// }
};
const subtitlesSchema = new mongoose.Schema(
	{
		like_count: {
			type: Number,
			default: 0,
		},
		age_limit: {
			type: Number,
			default : 0
		},
		duration: {
			type: Number,
			default: 0,
		},
		uploader_url: {
			type: String,
			required: true,
		},
		display_id: {
			type: String,
			required: true,
			
		},
		description: {
			type: String,
			default: '',
		},
		tags: {
			type: Array,
			default: []
		},
		title: {
			type: String,
			required: true,
		},
		channel_id: {
			type: String,
			required: true,
		},
		id: {
			type: String,
			required: true,
			index: { unique: true }
		},
		dislike_count: {
			type: Number,
			default: 0,
		},
		parsedSubtitles:{
			type: Array,
			default: null
		}


	},
	{ timestamps: {}, }
);
/**
 * Users model connection
 */
const users = async () => {
	// TODO: pull all the schemas that users uses: usersSchema, clipSchema and platformSchema

	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const usersModel = connection.model('User', usersSchema);
	return usersModel;
};

const subtitles = async () => {
	// TODO: pull all the schemas that users uses: usersSchema, clipSchema and platformSchema

	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const usersModel = connection.model('Subtitle', subtitlesSchema);
	return usersModel;
};

/**
 * Fortnite data model connection
 */
const fortniteData = async () => {
	// TODO: pull all the schemas that fortniteData uses: fortniteDataSchema and fortniteFrameSchema

	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const fortniteDataModel = connection.model('FortniteData', fortniteDataSchema, 'fortniteData');
	return fortniteDataModel;
};

/**
 * Statistics Dashboard model connection
 */
const healthChecks = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const healthChecksModel = connection.model('HealthChecks', statisticsDashboardSchema, 'healthChecks');
	return healthChecksModel;
};

/**
 * Fortnite data model connection
 */
const schedule = async () => {
	// TODO: pull all the schemas that fortniteData uses: fortniteDataSchema and fortniteFrameSchema

	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const scheduleModel = connection.model('Schedule', scheduleSchema, 'schedule');
	return scheduleModel;
};

const token = async () => {
	// TODO: pull all the schemas that fortniteData uses: fortniteDataSchema and fortniteFrameSchema

	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const tokenModel = connection.model('Token', tokenSchema, 'token');
	return tokenModel;
};

const chat = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const chatFrameModel = connection.model('Chat', chatFrameSchema, 'chatData');
	return chatFrameModel;
};

const games = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const gameModel = connection.model('Game', gameSchema);
	return gameModel;
};

const iterations = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const iterationModel = connection.model('Iteration', iterationSchema);
	return iterationModel;
};
const likes = async () => {
	if (!connection) {
		throw new Error('Connection was not initialized');
	}

	const iterationModel = connection.model('Like', likeSchema);
	return iterationModel;
};

/**
 * Added hooks to the streamer schema
 * @param {Object} streamerSchema
 * @param {Array} supportedGames
 */
const addStreamerSchemaHooks = (streamerSchema, supportedGames) => {
	// making sure unsupported games won't cause validation issues
	streamerSchema.pre('save', function (next) {
		if (this.currentGame) {
			if (!supportedGames.find((game) => game.name.toLowerCase() === this.currentGame.toLowerCase())) {
				this.currentGame = 'unsupported game';
			}
		}
		next();
	});

	streamerSchema.pre(/update/i, function (next) {
		if (this._update && this._update.currentGame) {
			if (!supportedGames.find((game) => game.name.toLowerCase() === this._update.currentGame.toLowerCase())) {
				this._update.currentGame = 'unsupported game';
			}
		}

		next();
	});
};

module.exports.Database = {
	init: init,
	streamers: streamers,
	users: users,
	fortniteData: fortniteData,
	healthChecks: healthChecks,
	schedule: schedule,
	token: token,
	chat: chat,
	games: games,
	iterations: iterations,
	likes: likes,
	subtitles: subtitles
};
