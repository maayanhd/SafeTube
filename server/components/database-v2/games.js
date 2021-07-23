const mongoose = require('mongoose');

exports.getGameSchema = (game) => {
	let baseGame = {
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
		teamSize: {
			// BASE
			type: Number,
			default: 1,
			min: 1,
		},
		statistics: {
			// BASE
			type: mongoose.Schema.Types.Mixed,
			default: {},
		},
	};

	// let gameSchema = {};

	// switch (game) {
	// 	case 'fortnite':
	// 		gameSchema = getFortniteSchema();
	// 		break;

	// 	default:
	// 		break;
	// }

	const finalGameSchema = new mongoose.Schema(baseGame, {
		_id: false,
	});

	return finalGameSchema;
};

// ! Ice Bucketed for now until we will use params for specific games
// // FORTNITE
// const getFortniteSchema = () => {
// 	return {
// 		firstGame: {
// 			type: Boolean,
// 			default: true,
// 		},
// 		eliminations: {
// 			type: Number,
// 			default: 0,
// 			min: 0,
// 		},
// 		timer: {
// 			type: Number,
// 			default: 0,
// 			min: 0,
// 		},
// 		remainingPlayers: {
// 			type: Number,
// 			default: 100,
// 			max: 100,
// 			min: 1,
// 		},
// 	};
// };
