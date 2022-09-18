conda activate ldh
workflow launch_all {
    parallel {
        Start-Process -FilePath "python" -ArgumentList "./scripts/sdb.py --modeltype normal --workers 1"
        Start-Process -FilePath "python" -ArgumentList "./scripts/sdb.py --modeltype waifu --workers 2"
        Start-Process -FilePath "python" -ArgumentList "./scripts/sdb.py --modeltype trinart2-min --workers 1"
        Start-Process -FilePath "python" -ArgumentList "./scripts/sdb.py --modeltype trinart2-max --workers 1"           
    }
}

launch_all
