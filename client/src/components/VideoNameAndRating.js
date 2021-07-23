import React, { useEffect, useState } from 'react';
import Tooltip from '@material-ui/core/Tooltip';
import { makeStyles } from '@material-ui/core/styles';
import IconButton from '@material-ui/core/IconButton';
import RatingVisual from './RatingVisual.js';



const useStyles = makeStyles({
    root: {
        "&:hover": {
            backgroundColor: "transparent"
        }
    }
});

const VideoNameAndRating = ({ videoDetailsDisplay, videoPlaylist, currentVideoIndexInPlaylist }) => {

    const [currentVideoName, setCurrentVideoName] = useState("");
    const [currentVideoRating, setCurrentVideoRating] = useState("");
    const [ratingVisual, setRatingVisual] = useState("none");
    const [iconTitle, setIconTitle] = useState("");
    const classes = useStyles();

    useEffect(() => {
        if (ratingVisual === "greenRatingVisual") {
            setIconTitle("Video is suitable for kids.");
        } else if (ratingVisual === "orangeRatingVisual") {
            setIconTitle("Video may contain language not suitable for kids.");
        }
    }, [ratingVisual])
    useEffect(() => {
        console.log(videoPlaylist.length > currentVideoIndexInPlaylist);
        if (videoPlaylist.length > currentVideoIndexInPlaylist) {
            setCurrentVideoName(videoPlaylist[currentVideoIndexInPlaylist].videoName);
            setCurrentVideoRating(videoPlaylist[currentVideoIndexInPlaylist].rating);
            if (videoPlaylist[currentVideoIndexInPlaylist].rating > 8.5) {
                setRatingVisual("greenRatingVisual");
            } else {
                if (videoPlaylist[currentVideoIndexInPlaylist].rating > 6.5) {
                    setRatingVisual("orangeRatingVisual");
                }
            }
        }
    }, [videoPlaylist, currentVideoIndexInPlaylist]);


    return (
        <div className={videoDetailsDisplay} >
            <span className="left">
                <h4>
                    {currentVideoName}
                </h4>
            </span>
            <span className="right">
                <h4>
                    {"Rating: " + currentVideoRating}
                </h4>
                <Tooltip title={iconTitle}>
                    <IconButton aria-label="delete" color="primary" className={classes.root} size="small">
                        <RatingVisual ratingVisual={ratingVisual} />
                    </IconButton>
                </Tooltip>
            </span>
        </div >
    );
}

export default VideoNameAndRating;