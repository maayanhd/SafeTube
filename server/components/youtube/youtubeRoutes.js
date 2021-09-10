const express = require('express');
const router = express.Router();
const { upsertYoutubeStreamers, getAllTranscripts, getTranscriptEndpoint, fetchYoutubePlaylist, fetchYoutubeVideo } = require('.');

router.route('/upsert').get(upsertYoutubeStreamers);//
router.route('/getalltranscripts').get(getAllTranscripts);//
router.route('/gettranscript').get(getTranscriptEndpoint);//
router.route('/playlist').get(fetchYoutubePlaylist);//
router.route('/video').get(fetchYoutubeVideo);//

// router.route('/update').post(updateStreamers);





module.exports = router;
