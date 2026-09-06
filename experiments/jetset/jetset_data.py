import numpy as np

_GLOBAL_TRACKS = None
_GLOBAL_JETS = None
_GLOBAL_TRUTH = None


#def read_single_jet_from_jetset(jet_idx, tracks=_GLOBAL_TRACKS, jets=_GLOBAL_JETS, truth=_GLOBAL_TRUTH,min_valid_tracks=3) :
def read_single_jet_from_jetset(jet_idx, tracks, jets, truth, min_valid_tracks=3):
    '''
    Process one jet from JetSet dataset.
    simulated top quark pair production at a centre-of-mass energy of 13.6 TeV provided by ATLAS collaboration
    https://opendata.cern.ch/record/93940
    https://gitlab.cern.ch/atlas/open-data/transforming-jet-flavor/-/blob/main/vars_open.md?ref_type=heads
    :param jet_idx:
    :param tracks:
    :param jets:
    :param truth:
    :return:
    '''

    # Make sure to only grab valid particle tracks in the jet and non-padded ones
    track_valid = tracks['valid'][jet_idx].astype(bool)

    # If no tracks are valid, return None instead of an empty dict
    if int(np.sum(track_valid)) < min_valid_tracks:
        return None

    # track level variables
    track_deta = tracks['deta'][jet_idx][track_valid] #pseudorapidity distance between track and jet
    track_dphi = tracks['dphi'][jet_idx][track_valid] #azimuthal angle distance between track and jet
    track_ptfrac = tracks['ptfrac'][jet_idx][track_valid] #fraction of jet transverse momentum p_T carries by the track
    track_pt = tracks['pt'][jet_idx][track_valid] #track transverse mpmentum p_T: momentum of the track perpendicular to the beam pipe
    # transverse impact parameter is the shortest distance between the track and the PV (e.g. d0~0, particle is probs straight fomr collision)
    track_d0 = tracks['d0'][jet_idx][track_valid] #transverse impact parameter relative to Primary Vertex
    # distance between the track and the PV along the beam axis (z), but it is "scaled" by sin\theta to make the measurement independent of the angle at which the particle was produced
    track_z0SinTheta = tracks['z0SinTheta'][jet_idx][track_valid] #longitudinal impact parameter projected onto the direction perpendicular to the track, relative to the PV
    track_lifetimeSignedD0Significance = tracks['lifetimeSignedD0Significance'][jet_idx][track_valid]
    # lifetime-signed track d0 significance
    # If a track comes from a particle that traveled a bit before decaying (like a B-hadron), its track will likely point ``away" from the PV in the same direction as the jet
    # + sign means the track crosses the jet axis in front of PV (suggest real decay travel distance)
    # - sign means track crosses behind PB so likely noise
    track_lifetimeSignedD0 = tracks['lifetimeSignedD0'][jet_idx][track_valid] # lifetime-signed transverse impact parameter
    track_lifetimeSignedZ0SinTheta = tracks['lifetimeSignedZ0SinTheta'][jet_idx][track_valid] # lifetime-signed longitudinal impact parameter, multiplied by \sin\theta.
    track_ftagTruthVertexIndex = tracks['ftagTruthVertexIndex'][jet_idx][track_valid] # Truth vertex index of the track. 0 is reserved for the truth PV, others SV

    # track truth labels: 0 = pileup, 1 = fake, 2 = prompt, 3 = from b-hadron, 4 = from c child of a bhadron, 5 = from c-hadron, 6 = from \tau  lepton, 7 = from other secondary decay
    track_ftagTruthOriginLabel = tracks['ftagTruthOriginLabel'][jet_idx][track_valid] # Truth origin of the track


    # Jet level variables
    jet_pt = float(jets['pt'][jet_idx])  # total momentum of all particles in the cone
    jet_eta = float(jets['eta'][jet_idx])  # the central axis of the cluster of particles.
    jet_phi = float(jets['phi'][jet_idx])  # the average direction of the entire cluster of particles.
    # Jet label, using a geometric cone around the jet rather than the jet clustering algorithm.
    # If a parton with a transverse momentum of more than 5 GeV is found within \delta R(q, jet) < 0.3 of the jet direction,
    # the jet is labelled as a jet with the parton's flavor.
    # The label should be one of: 0 (light jet), 4 (charm jet), 5 (bottom jet), or 15 (tau jet).
    jet_label = int(jets['HadronConeExclTruthLabelID'][jet_idx])


    # Truth hadron variables
    truth_valid = truth['valid'][jet_idx].astype(bool)
    # Transverse Decay Length is the distance in the x−y plane from the PV to the point where the particle decayed
    # So this is the actual physical distance that the b-hadron traveled before decaying
    # High Lxy is the signature of long-lived particles like b-quarks or c-quarks.
    truth_Lxy = truth['Lxy'][jet_idx][truth_valid]
    # spatial coordinates of where where decay happened
    truth_decayVertexZ = truth['decayVertexZ'][jet_idx][truth_valid] # Z component of the truth particle decay displacement
    truth_decayVertexDPhi = truth['decayVertexDPhi'][jet_idx][truth_valid] #Truth particle decay \phi offset, relative to the jet axis
    truth_pdgId = truth['pdgId'][jet_idx][truth_valid] #Truth particle pdgId

    # Physical substructure observables
    num_tracks = int(track_valid.sum()) # counts number of tracks
    num_sv_tracks = int(np.sum(track_ftagTruthVertexIndex > 0))  # counts SV tracks

    frac_sv = num_sv_tracks / num_tracks if num_tracks > 0 else 0

    num_vertices = int(len(np.unique(track_ftagTruthVertexIndex))) # number of unique vertices
    num_sv_vertices = int(len(np.unique(track_ftagTruthVertexIndex[track_ftagTruthVertexIndex > 0])))  # number of unique SV

    # Calculate radial distance for each track from jet axis
    track_dR = np.sqrt(track_deta**2 + track_dphi**2)

    # p_T-weighted jet width (standard substructure observable)
    jet_width = float(np.sum(track_ptfrac * track_dR))
    # Maximum impact significance in the jet :
    # Significance tells us how many "standard deviations" away a track is from the primary interaction point
    max_d0_sig = float(np.max(np.abs(track_lifetimeSignedD0Significance))) if num_tracks > 0 else 0

    # Use the truth Lxy (decay displacement) as the main physical correlate
    max_Lxy = float(np.max(truth_Lxy)) if len(truth_Lxy) > 0 else 0
    # Track origins: 3,4 = B-hadron chain; 5 = C-hadron
    frac_from_b = float(np.mean((track_ftagTruthOriginLabel == 3) | (track_ftagTruthOriginLabel == 4)))
    frac_from_c = float(np.mean(track_ftagTruthOriginLabel == 5))


    jet_data = {
        'jet': {
            'pt': jet_pt,
            'eta': jet_eta,
            'phi': jet_phi,
            'label': jet_label,
        },
        'tracks': {
            'angular_coord': np.stack([track_deta, track_dphi], axis=0),
            'impact_parameter_coord': np.stack([track_d0, track_z0SinTheta], axis=0),
            'lifetime_coord': np.stack([track_lifetimeSignedD0, track_lifetimeSignedZ0SinTheta], axis=0),
            'deta': track_deta,
            'dphi': track_dphi,
            'ptfrac': track_ptfrac,
            'pt': track_pt,
            'dR': track_dR,
            'track_vertex_id': track_ftagTruthVertexIndex, #Truth vertex index of the track. 0 is reserved for the truth PV, others SV
            'origin_label': track_ftagTruthOriginLabel, # Truth origin of the track
        },
        #properties of the hadrons that existed before they decayed
        'truth_particles':{
            'Lxy': truth_Lxy,
            'decayVertexZ': truth_decayVertexZ,
            'decayVertexDPhi': truth_decayVertexDPhi,
            'pdgId': truth_pdgId,
        },
        'physical_substructure':{
            'num_tracks': num_tracks,
            'num_sv_tracks': num_sv_tracks,
            'frac_sv': frac_sv,
            'num_vertices': num_vertices,
            'num_sv_vertices': num_sv_vertices,
            'jet_width': jet_width,
            'frac_from_b': frac_from_b,
            'frac_from_c': frac_from_c,
        },
        'global_truth':{
            'max_Lxy': max_Lxy,
            'max_d0_sig': max_d0_sig,
        }

    }
    return jet_data



