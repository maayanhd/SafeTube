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

const VideoViewsAndLikes = ({ videoDetailsDisplay, views, likes, dislikes }) => {
    const [likeAmount, setLikeBarAmount] = useState(likes);
    const [disLikeAmount, setDisLikeBarAmount] = useState(dislikes);


    let likeBarAmount = ((likes) / (likes + dislikes)) * 150;
    let disLikeBarAmount = 150 - likeBarAmount;

    const classes = useStyles();

    return (
        <div className={videoDetailsDisplay} >
            <span className="left">
                <h4>
                    {views + " Views"}
                </h4>
            </span>
            <span className="right">
                <h4>
                    <Tooltip title={"Likes: " + likeAmount}>
                        <IconButton aria-label="delete" color="primary" className={classes.root} size="small">
                            <div className="likeBar" style={{ width: likeBarAmount }}></div>
                        </IconButton>
                    </Tooltip>
                    <Tooltip title={"Dislikes: " + disLikeAmount}>
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