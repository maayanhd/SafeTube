import React, { useState } from 'react';
import Tooltip from '@material-ui/core/Tooltip';
import { makeStyles } from '@material-ui/core/styles';
import IconButton from '@material-ui/core/IconButton';

const useStyles = makeStyles({
    root: {
        "&:hover": {
            backgroundColor: "transparent",

        },
        padding: '0 0px'
    }
});

const VideoViewsAndLikes = ({ videoDetailsDisplay, currVideoViews, currVideoLikes, currVideoDislikes }) => {


    let likeBarAmount = ((currVideoLikes) / (currVideoLikes + currVideoDislikes)) * 150;
    let disLikeBarAmount = 150 - likeBarAmount;

    const classes = useStyles();

    return (
        <div className={videoDetailsDisplay} >
            <span className="left">
                <h4>
                    {currVideoViews + " Views"}
                </h4>
            </span>
            <span className="right">
                <h4>
                    <Tooltip title={"Likes: " + currVideoLikes}>
                        <IconButton aria-label="delete" color="primary" className={classes.root} size="small">
                            <div className="likeBar" style={{ width: likeBarAmount }}></div>
                        </IconButton>
                    </Tooltip>
                    <Tooltip title={"Dislikes: " + currVideoDislikes}>
                        <IconButton aria-label="delete" color="primary" className={classes.root} size="small">
                            <div className="disLikeBar" style={{ width: disLikeBarAmount }}></div>
                        </IconButton>
                    </Tooltip>
                </h4>
            </span>
        </div >
    )

}

export default VideoViewsAndLikes;