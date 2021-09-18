const { Database } = require('../database-v2');
const { google } = require('googleapis');
const asyncHandler = require('../backend-async-handler');
const logger = require('../logger');
const URL = require('url').URL;
const youtubedl = require('youtube-dl-exec')
const fetch = require('node-fetch');
const xmlParser = require('xml-js');

// const initYouTubeAPI = () => {
// 	return google.youtube({
// 		version: 'v3',
// 		auth: 'AIzaSyCWQYCbkfQfSRxvakEBpqM8Z8CN2kD1Qvk',
// 	});
// };

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
	const subtitlesKeys = Object.keys(subtitles)
	let res = {};
	// iterate over each keys of source
	whiteList.forEach((key) => {
	  // if whiteList contains the current key, add this key to res
	  if (key != "thumbnail" || subtitlesKeys.indexOf(key) !== -1) {
		res[key] = subtitles[key];
	  }
	});
	//Here
	console.log("trying to get scoring");
	// let scoring = {};
	// if (subtitles.parsedSubtitles.length == 0) {
    //     scoring = {
    //         "bracket_count": 0,
    //         "contains_bad_language": true,
    //         "is_safe": false,
    //         "final_score": 2
    //     }
	// 	res.scoring = scoring;
	// } else {
	// 	scoring  = await getScoring(subtitles);
	// 	console.log(scoring);
	// 	if(scoring){
	// 		res.scoring = scoring;
	// 	}
	// }
	
	res.scoring = await getScoring(subtitles);
	console.log("res.scoring");
	console.log(res.scoring);
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
	// const upsertYoutubeStreamers = asyncHandler(async (req, res) => {//
const upsertYoutubeStreamers = async (inputUrl) => {//
	const subtitleModel = await Database.subtitles();
	let videoData = {};
	
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
		// console.log("videoData\n");
		// console.log(videoData);
	} catch (error) {
		console.log(error.stderr);
		return;

	}
	let video = await subtitleModel.findOne({id : videoData.id});
	
	//Video not found in DB => parse subtitles and save to DB
	if (video == null) {
		try {
			let urls = null;
			videoData.parsedSubtitles = [];		
			if(videoData.subtitles || videoData.automatic_captions){
				try {
					urls = videoData.subtitles.en ?? videoData.automatic_captions.en ?? null;
				} catch (error) {
					console.log("no english subtitles");
				}
			}
			if(urls){
				urls = urls.filter(e=>e.ext.includes('srv'));
				const firstUrl = urls[0].url;
				try {
					let captionJson = await fetch(firstUrl).then(res => res.text()).then(res=> {
						// console.log("debug\n\n\n\n");
						// console.log(res);
						return xmlParser.xml2json(res, {compact: true, spaces: 4})
					})
					captionJson = JSON.parse(captionJson);
					captionJson = cleanSubtitles(captionJson);
					videoData.parsedSubtitles = captionJson;
				} catch (error) {
					// console.log(error);
					videoData.parsedSubtitles = [];					
				}
		
			}
			video = await saveSubtitlesToDB(videoData);
		} catch (error) {
			console.log(error);
			return;
		}
	}
	// video["channelScore"] =  await appendRatePerChannel(video);
	return video;
};
const getScoring = async (videoData) =>{
	let result = {
		"bracket_count": 0,
		"contains_bad_language": true,
		"is_safe": false,
		"final_score": 2
	};
	if (videoData.parsedSubtitles && videoData.parsedSubtitles.length > 0) {
 		const url = "http://localhost:8081";
		try {
			await fetch(url, {
			method: 'POST',
			body: JSON.stringify(videoData),
			headers: { 'Content-Type': 'application/json' }
			}).then(res => res.text()).then(json => {
				try {
					result = JSON.parse(json)[0];
				} catch (error) {
					console.log("Bad result from scoring server\n" + error);
				}
			})	
		} catch (error) {
			console.log(error);			
		}
	}

	return result;
};

const getTranscripts = async (videoIdArray) =>{
	const subtitleModel = await Database.subtitles();
	if (videoIdArray.length == 1) {
		return subtitleModel.findOne({id : videoId}).select({id : 1, parsedSubtitles : 1});
	}

	return subtitleModel.find({id : { $in : videoIdArray}}).select({id : 1, parsedSubtitles : 1});
};
const getTranscriptEndpoint = asyncHandler(async (req, res) => {
	const videoId = req.query.videoid ?? null;
	if(!videoId){
		res.status(500).send('No videoId header field');
		return;
	}
	let result = await getTranscripts([videoId])
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
	//Verifing google api result is correct
	if (playlist.items && playlist.items.length > 0){
		result.status = 200
		result.data = playlist;
	}else{
		result.data = `No data found under the playlist ${currentPlaylistID}`;
		res.send(result);
		return;
	}

	let playlistVideoIds = playlist.items.map(e=>e.snippet.resourceId.videoId);
	let promiseArr = [];
	result = [];
	for (let index = 0; index < playlistVideoIds.length; index++) {
		const element = playlistVideoIds[index];
		promiseArr.push(upsertYoutubeStreamers(`https://www.youtube.com/watch?v=${element}`));
	}
	for (let index = 0; index < promiseArr.length; index+=9) {
		result.push(...await Promise.all(promiseArr.slice(index, index + Math.min(promiseArr.length-index,9) ) ) )
		
	}
	for (let index = 0; index < result.length; index++) {
		if (result[index]) {
			result[index]["channelScore"] =  await appendRatePerChannel(result[index]);
		}
		
	}
	
	// result = await Promise.all(promiseArr);
	// res.send(result);
	// console.log(result[0]);
	
	res.send(result);
	
	// console.log(result.map(x=>{
	// 	if(x){
	// 		return {
	// 			channelScore : x.channelScore,
	// 			display_id : x.display_id
	// 		}
	// 	}else{
	// 		return null
	// 	}
	// }));
	
});

const fetchYoutubeVideo = asyncHandler(async (req, res) => {
	const videoID = req.query.videoid ?? null;
	let result = {status : 500 , data : null};
	if(videoID) {
		upsertResult = await upsertYoutubeStreamers(`https://www.youtube.com/watch?v=${videoID}`);
		if(upsertResult){
			result = upsertResult; 
		}
	}
	res.send(result);
});

const appendRatePerChannel = async (videoObj) => { 
	const subtitleModel = await Database.subtitles();
	let channelVideos = await subtitleModel.find({channel_id : videoObj.channel_id});
	let goodVideoCount = channelVideos.filter((videoObj)=>{
		return videoObj.scoring.final_score >= 6;
	  }).length;
	return {
		goodVideoCount :  goodVideoCount,
		badVideoCount : channelVideos.length - goodVideoCount
    }
};

(async()=>{
	setTimeout(async () => {
		// urls = "blabla"
		// videoData = {
		// 	subtitles : {
		// 		dx : "sub en"
		// 	}
		// }
		// try {
		// 	urls = videoData.subtitles.en ?? videoData.automatic_captions.en ?? null;
		// } catch (error) {
		// 	console.log(error);
		// }
		// console.log(urls);


		// const subtitleModel = await Database.subtitles();
		// await subtitleModel.updateMany({'scoring' : null},{'scoring' : {
		// 	bracket_count : 0,
		// 	contains_bad_language : true,
		// 	is_safe : false,
		// 	final_score : 6
		// }})
		// let x = await subtitleModel.find({'scoring' : null});
		// // let x = await subtitleModel.find({'uploader' : { "$regex": "jxdn", "$options": "i" },"scoring.final_score" : {$gte : 6}});
		// console.log(x.length);


		// console.log(x.map(x=> x.id));
		// x = x.map(x => x.scoring)
		// await fetchYoutubePlaylist("https://www.youtube.com/watch?v=lVKk__uuIxo&list=PLF7tUDhGkiCk_Ne30zu7SJ9gZF9R9ZruE")
		// const foo = [1, 2, 3];
		// const [n] = foo;
		// console.log(n);
		// console.log(await getAllTranscripts());
		
		// console.log("starting");
		// await upsertYoutubeStreamers("https://www.youtube.com/watch?v=5sTWbG3n8wU");
		// initYouTubeAPI();
		// 
		// console.log("video");
		// const streamersModel = await Database.streamers();
		// let streamer = new streamersModel({name : "MoryZz",platform:{name : "twitch",id: 1234}})
		// await streamer.save();
		// x = [1,2,3,4,5,6,7,8,9,10,11,12]
		// result = []
		// for (let index = 0; index < x.length; index+=9) {
		// 	result.push(...x.slice(index, index + Math.min(x.length-index,9) ) )
			
		// }
		// console.log(result);
		// x.slice(2,4).map(x=>x+1)
		// console.log(x.slice(2,4));
		// console.log(x);

	}, 1000);
	
})();

module.exports = {
	upsertYoutubeStreamers: upsertYoutubeStreamers,
	getAllTranscripts: getAllTranscripts,
	getTranscriptEndpoint: getTranscriptEndpoint,
	fetchYoutubePlaylist: fetchYoutubePlaylist,
	fetchYoutubeVideo: fetchYoutubeVideo
	
};