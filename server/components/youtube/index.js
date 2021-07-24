const { Database } = require('../database-v2');
const { google } = require('googleapis');
const asyncHandler = require('../backend-async-handler');
const logger = require('../logger');
const URL = require('url').URL;
const youtubedl = require('youtube-dl-exec')
const fetch = require('node-fetch');
const xmlParser = require('xml-js');




const { exit } = require('process');
const { count } = require('console');
const { default: youtubeDlExec } = require('youtube-dl-exec');
const { json } = require('express');

const initYouTubeAPI = () => {
	return google.youtube({
		version: 'v3',
		auth: 'AIzaSyCWQYCbkfQfSRxvakEBpqM8Z8CN2kD1Qvk',
	});
};

/**
 * Handles online & offline streamers iteration number and schedule according to status
 * @param {Number} updateIteration the last update iteration
 * @param {String} platformName
 */
const cleanSubtitles = (subtitles)=>{

	let text = subtitles.transcript ?? null;
	if(text){
		text = text.text
		text = text.map(e=>e._text)
	}
	return text;
}
/**
 * Handles online & offline streamers iteration number and schedule according to status
 * @param {Number} updateIteration the last update iteration
 * @param {String} platformName
 */
const saveSubtitlesToDB = asyncHandler(async (subtitles)=>{
	const subtitleModel = await Database.subtitles();
	const whiteList = Object.keys(subtitleModel.schema.obj);
	let res = {};
	// iterate over each keys of source
	Object.keys(subtitles).forEach((key) => {
	  // if whiteList contains the current key, add this key to res
	  if (whiteList.indexOf(key) !== -1) {
		res[key] = subtitles[key];
	  }
	});
	res = new subtitleModel(res);
	await res.save()
	console.log('saved');
	return res;
  
});
/**
 * Handles online & offline streamers iteration number and schedule according to status
 * @param {Number} updateIteration the last update iteration
 * @param {String} platformName
 */
	const upsertYoutubeStreamers = asyncHandler(async (req, res) => {//
// const upsertYoutubeStreamers = asyncHandler(async (inputUrl) => {//
	const subtitleModel = await Database.subtitles();
	let videoData = {}
	try {
	
		videoData = await youtubedl(inputUrl, {
			dumpJson: true,
			noWarnings: true,
			noCallHome: true,
			noCheckCertificate: true,
			preferFreeFormats: true,
			youtubeSkipDashManifest: true,
			referer: inputUrl
		})

	} catch (error) {
		console.log("123\n");
		console.log(error.stderr);
		if(error.stderr.includes("inappropriate")){
			saveSubtitlesToDB({})
		}
		throw(error.stderr);
		console.log("123\n");

	}

	let urls = null;
	if(videoData.subtitles){
		urls = videoData.subtitles.en ?? videoData.automatic_captions.en ?? null;
	}
	console.log(urls);
	// // // ! DONE
	if(urls){
		urls = urls.filter(e=>e.ext.includes('srv'));
		const firstUrl = urls[0].url;
		let captionJson = await fetch(firstUrl).then(res => res.text()).then(res=> xmlParser.xml2json(res, {compact: true, spaces: 4}))
		captionJson = JSON.parse(captionJson);
		captionJson = cleanSubtitles(captionJson);
		videoData.parsedSubtitles = captionJson;

	}
	// else{
	// 	res.send('Somthing went wrong with fetching the urls');
	// 	return;
	// }
	
	// ! DONE
	// console.log(videoData);
	let blabla = await subtitleModel.findOne({id : videoData.id});
	console.log(blabla);
	if (!blabla) {
		try {
			await saveSubtitlesToDB(videoData);
		} catch (error) {
			console.log(error);
			res.send('Somthing went wrong with saving to DB');
			return;
		}
	}
	
	res.send('OK!');
	// res.send(videoData);
	return;
});

const getTranscript = asyncHandler(async (req, res) => {
	const videoId = req.query.videoid ?? null;
	if(!videoId){
		res.status(500).send('No videoId header field');
		return;
	}
	const subtitleModel = await Database.subtitles();
	let result = await subtitleModel.findOne({id : videoId}).select({id : 1, parsedSubtitles : 1});
	if(result == null){
		res.status(500).send(`No video found by the ID ${videoId}`)
		return;
	}
	res.send(result);
});
const getAllTranscripts = asyncHandler(async (req, res) => {
	const subtitleModel = await Database.subtitles();
	let result = await subtitleModel.find({}).select({id : 1, parsedSubtitles : 1});
	res.send(result);
	return result;
});

const fetchYoutubePlaylist = asyncHandler(async (req, res) => {
	let result = {status : 500 , data : null};
	const resultsFromPlaylist = 50;
	const currentPlaylistID = req.query.playlistid ?? null;
	 if(!currentPlaylistID){
		 result.data = 'No playlist header field'
		 res.send(result);
		 return;
	 }
	const playlist = await fetch(
		"https://www.googleapis.com/youtube/v3/playlistItems?maxResults=" + resultsFromPlaylist + "&playlistId=" + currentPlaylistID + "&part=snippet&fields=items%2Fid%2C%20items%2Fsnippet(title%2Cdescription%2CvideoOwnerChannelTitle%2Cthumbnails(medium)%2CresourceId)&key=AIzaSyD5aoCjGplRER2cPriT28Osh7McTSW6QDk",
		{ method: "GET" })
	.then(respones => respones.json());
	if (playlist.items && playlist.items.length > 0){
		result.status = 200
		result.data = playlist;
	}else{
		result.data = `No data found under the playlist ${currentPlaylistID}`;
	}
	res.send(result);
});
(async()=>{
	setTimeout(async () => {
		// await fetchYoutubePlaylist("https://www.youtube.com/watch?v=lVKk__uuIxo&list=PLF7tUDhGkiCk_Ne30zu7SJ9gZF9R9ZruE")
		// const foo = [1, 2, 3];
		// const [n] = foo;
		// console.log(n);
		// console.log(await getAllTranscripts());
		
		// console.log("starting");
		// upsertYoutubeStreamers("https://www.youtube.com/embed/gEX_RS3_IzI");
		// initYouTubeAPI();
		// 
		// console.log("blabla");
		// const streamersModel = await Database.streamers();
		// let streamer = new streamersModel({name : "MoryZz",platform:{name : "twitch",id: 1234}})
		// await streamer.save();
	}, 10000);
	
})();

// updateStreamers: updateStreamers,
module.exports = {
	upsertYoutubeStreamers: upsertYoutubeStreamers,
	getAllTranscripts: getAllTranscripts,
	getTranscript: getTranscript,
	fetchYoutubePlaylist: fetchYoutubePlaylist
	
};
