const express = require('express');
const router = express.Router();
const { upsertYoutubeStreamers, getAllTranscripts, getTranscript, fetchYoutubePlaylist } = require('.');

router.route('/upsert').get(upsertYoutubeStreamers);//
router.route('/getalltranscripts').get(getAllTranscripts);//
router.route('/gettranscript').get(getTranscript);//
router.route('/playlist').get(fetchYoutubePlaylist);//

// router.route('/update').post(updateStreamers);





module.exports = router;
