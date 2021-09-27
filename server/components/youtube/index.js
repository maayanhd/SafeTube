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
const saveSubtitlesToDB = asyncHandler(async (subtitlesArr)=>{
	const subtitleModel = await Database.subtitles();
	const whiteList = Object.keys(subtitleModel.schema.obj);
	let resArr = [];
	let resSavePromiseArr = [];
	for (let index = 0; index < subtitlesArr.length; index++) {
		const subtitles = subtitlesArr[index];
		const subtitlesKeys = Object.keys(subtitles)
		let res = {};
		// iterate over each keys of source
		whiteList.forEach((key) => {
		  // if whiteList contains the current key, add this key to res
		  if (key != "thumbnail" || subtitlesKeys.indexOf(key) !== -1) {
			res[key] = subtitles[key];
		  }
		});
		res = new subtitleModel(res);
		resSavePromiseArr.push(res.save());
		console.log('saved');
		resArr.push(res);
		
	}
	await Promise.all(resSavePromiseArr);
	return resArr;
	
});
/**
 * Handles online & offline streamers iteration number and schedule according to status
 * @param {Number} updateIteration the last update iteration
 * @param {String} platformName
 */
	// const upsertYoutubeStreamers = asyncHandler(async (req, res) => {//
const upsertYoutubeStreamers = async (inputUrl,display_id) => {//
	let videoData = {};
	//Video not found in DB => parse subtitles and save to DB
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
			//video = await saveSubtitlesToDB(videoData);
		} catch (error) {
			console.log(error);
		}
	
	// video["channelScore"] =  await appendRatePerChannel(video);
	return videoData;
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
const newGetScoring = async (videosData) =>{
	const url = "http://localhost:8081";
	let result = videosData.map(x=>{
		return {
			id:x.id,
			"bracket_count": 0,
			"contains_bad_language": true,
			"is_safe": false,
			"final_score": 2
		}
	});
	try {
		await fetch(url, {
		method: 'POST',
		body: JSON.stringify(videosData),
		headers: { 'Content-Type': 'application/json' }
		}).then(res => res.text()).then(json => {
			try {
				result = JSON.parse(json);
			} catch (error) {
				console.log("Bad result from scoring server\n" + error);
			}
		})	
	} catch (error) {
		console.log(error);			
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
	const subtitleModel = await Database.subtitles();
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
	let videoDataPromiseArr = [];
	result = [];
	
	for (let index = 0; index < playlistVideoIds.length; index++) {
		const element = playlistVideoIds[index];
		let video = await subtitleModel.findOne({id : element});
		if (video) {
			console.log("video found in the DB");
			result.push(video);
		} else {
			console.log("video currently not in the DB");
			videoDataPromiseArr.push(upsertYoutubeStreamers(`https://www.youtube.com/watch?v=${element}`,element));
		}
	}
	if (videoDataPromiseArr.length > 0) {
		console.log("in hereeeeeeeeeee");
		videoDataArr = await Promise.all(videoDataPromiseArr);
		videoDataArr = await getMultipleSubtitleScoring(videoDataArr);
		videoDataArr = await saveSubtitlesToDB(videoDataArr);
		result.push(...videoDataArr);
	}

	result = await fetchAndAttachChannelScoring(result);

	res.send(result);
});
const getMultipleSubtitleScoring = async (videoDataArr)=>{
	
	let subsAndIds = videoDataArr.map(video => {
		return {
			parsedSubtitles:video.parsedSubtitles,
			id:video.id
		}
	})
	let multiScoring = await newGetScoring(subsAndIds);
	videoDataArr.forEach(video => {
		let scoring = null;
		try {
			scoring = multiScoring.find(x=>x.id == video.id);
		} catch (error) {
			console.log(error);			
		}
		if (scoring) {
			video.scoring = scoring;
		}

	});

	return videoDataArr
	
}
const fetchAndAttachChannelScoring = async(videoObjArr)=>{
	const subtitleModel = await Database.subtitles();
	let uniqeChannelIdArr = [...new Set(videoObjArr.map(x=>x.channel_id))]
	for (let index = 0; index < uniqeChannelIdArr.length; index++) {
		if (uniqeChannelIdArr[index]) {
			uniqeChannelIdArr[index] = appendRatePerChannel(uniqeChannelIdArr[index],subtitleModel);
		}
		
	}
	uniqeChannelIdArr = await Promise.all(uniqeChannelIdArr);

	for (let index = 0; index < uniqeChannelIdArr.length; index++) {
		const uniqeChannelScoring = uniqeChannelIdArr[index];
		videoObjArr.map(x=>{
			if (x.channel_id == uniqeChannelScoring.channel_id) {
				x["channelScore"] = uniqeChannelScoring;
			}
			return x;
		})
		
	}
	return videoObjArr;
};


const fetchYoutubeVideo = asyncHandler(async (req, res) => {
	const videoID = req.query.videoid ?? null;
	let result = {status : 500 , data : null};
	if(videoID) {
		upsertResult = [await upsertYoutubeStreamers(`https://www.youtube.com/watch?v=${videoID}`,videoID)];
		upsertResult = await getMultipleSubtitleScoring(upsertResult);
		upsertResult = await saveSubtitlesToDB(upsertResult);
		upsertResult = await fetchAndAttachChannelScoring(upsertResult)
		if(upsertResult){
			result = upsertResult; 
		}
	}
	res.send(result);
});

const appendRatePerChannel = async (channel_id,subtitleModel) => { 
	let channelVideos = await subtitleModel.find({channel_id : channel_id});
	let goodVideoCount = channelVideos.filter((videoObj)=>{
		return videoObj.scoring.final_score >= 6;
	  }).length;
	return {
		channel_id : channel_id,
		goodVideoCount :  goodVideoCount,
		badVideoCount : channelVideos.length - goodVideoCount
    };
};


module.exports = {
	upsertYoutubeStreamers: upsertYoutubeStreamers,
	getAllTranscripts: getAllTranscripts,
	getTranscriptEndpoint: getTranscriptEndpoint,
	fetchYoutubePlaylist: fetchYoutubePlaylist,
	fetchYoutubeVideo: fetchYoutubeVideo
	
};