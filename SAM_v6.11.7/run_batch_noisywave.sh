#!/bin/bash

damp=2
noiselevel=1.0
group=2
for wn in {6..20}; do
    
    #caseid="256ens_damp${damp}day_sub45step_noadvectbg_1_"
    caseid="damp${damp}day_noadvectbg_noiselevel_${noiselevel}_${group}_"
    cp SAM_ADV_MPDATA_SGS_TKE_RAD_RRTM_MICRO_SAM1MOM_RCE_noisywave SAM_ADV_MPDATA_SGS_TKE_RAD_RRTM_MICRO_SAM1MOM_RCE_noisywave_wn${wn}_${caseid}
    
    DEST_DIR="RCE_noisywave_wn${wn}"
    SOURCE_DIR="RCE_noisywave_base"
    
    # Check if the destination directory exists
    if [ ! -d "$DEST_DIR" ]; then
        echo "Directory $DEST_DIR does not exist. Copying from $SOURCE_DIR..."
        # Copy the source directory to the destination
        cp -r "$SOURCE_DIR" "$DEST_DIR" 
    else
        echo "Directory $DEST_DIR already exists."
    fi
    cd $DEST_DIR
    echo "RCE_noisywave_wn${wn}" > CaseName
    
    PRM_FILE="../RCE_noisywave_base/prm"
    NEW_PRM_FILE="prm.${caseid}"
    cp "$PRM_FILE" "$NEW_PRM_FILE"
    # Modify the new prm file
    sed -i "3s|.*|caseid = '${caseid}'|" "$NEW_PRM_FILE"
    sed -i "s/^caseid_restart.*/caseid_restart = 'spinup_${group}_'/" "$NEW_PRM_FILE"
    sed -i "s/^wavenumber_factor.*/wavenumber_factor = $(echo "scale=3; $wn / 40" | bc)/" "$NEW_PRM_FILE"
    sed -i "s/^wavedampingtime.*/wavedampingtime = $damp/" "$NEW_PRM_FILE"
    sed -i "s/^wavetqdampingtime.*/wavetqdampingtime = $damp/" "$NEW_PRM_FILE"
    sed -i "s/^noiselevel.*/noiselevel = $noiselevel/" "$NEW_PRM_FILE"
    
    RESUB_FILE="resub_${group}.ens"
    cp "../RCE_noisywave_base/resub.ens" "$RESUB_FILE"
    sed -i "s/^#SBATCH -J.*/#SBATCH -J RCE_nw_${wn}/" "$RESUB_FILE"
    sed -i "s/^case=.*/case=RCE_noisywave_wn${wn}/" "$RESUB_FILE"
    sed -i "s/^subcase=.*/subcase=${caseid}/" "$RESUB_FILE"
    
    sleep 5

    #if (( $wn % 1 == 0 )); then
    #    sbatch "$RESUB_FILE"
    #fi

    #sleep 5
    
    cd ..
    
done
