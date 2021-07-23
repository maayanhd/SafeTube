const express = require('express');
const router = express.Router();
const { upsertYoutubeStreamers, updateStreamers } = require('.');

router.route('/upsert').get(upsertYoutubeStreamers);//
// router.route('/update').post(updateStreamers);

module.exports = router;
