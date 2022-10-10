

#######################
# VISUALIZATION UTILS #
#######################
from collections import defaultdict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from gym_minigrid.minigrid import DIR_TO_VEC, TILE_PIXELS, Goal, Key
from IPython.core.display import display
from PIL import Image
from scipy.interpolate import splev, splprep

from uniMASK.distributions import Categorical
from uniMASK.envs.minigrid.agents import StochGoalAgent
from uniMASK.envs.minigrid.data import r_idx_to_r, rtg_idx_to_rtg
from uniMASK.envs.minigrid.env import CustomActions, CustomDoorKeyEnv6x6, CustomDoorKeyEnv16x16
from uniMASK.utils import format_str_to_red

AGENT_DIR_TO_STR = {
    CustomActions.RIGHT: "➡️",
    CustomActions.DOWN: "⬇️",
    CustomActions.LEFT: "⬅️",
    CustomActions.UP: "⬆️",
}


def display_agent(env, agent_pos=None, agent_dir=None, door_open=None, key_pos=None, key_picked=None):
    """
    Display the agent at a desired position. Is currently mostly a hack to be able to create arbitrary visualizations
    """
    if env.base_env.__class__ in [CustomDoorKeyEnv6x6, CustomDoorKeyEnv16x16]:

        key = env.get_key()
        if key is not None:
            x, y = key.cur_pos
            env.grid.set(x, y, None)

        door = env.get_door()
        door.is_open = door_open
        if not key_picked and key_pos is not None:
            if not isinstance(key_pos, tuple):  # hack
                key_pos = env.IDX_TO_POS[key_pos.item()]
            key = Key("red")
            key.cur_pos = key_pos
            key.init_pos = key_pos
            env.grid.set(key_pos[0], key_pos[1], key)

        if agent_pos is not None:
            agent_pos = tuple(list(agent_pos))
            if agent_dir is None:
                agent_dir = env.agent_dir

    return env.grid.render(TILE_PIXELS, agent_pos=agent_pos, agent_dir=agent_dir)


def visualize_input_data(env, data, traj_idx=0):
    """
    Visualizes a FullTokenSeq
    """
    state_traj = data.get_factor("state").inputs_hr[traj_idx]
    act_traj = data.get_factor("action").inputs_hr[traj_idx]
    rew_traj = data.get_factor("reward").inputs_hr[traj_idx]
    rtg_traj = data.get_factor("rtg").inputs_hr[traj_idx]

    if env.base_env.__class__ in [CustomDoorKeyEnv6x6, CustomDoorKeyEnv16x16]:
        door_open = data.get_factor("state_door").inputs_hr[traj_idx]
        key_pos = data.get_factor("state_key_pos").inputs_hr[traj_idx]
        key_picked = data.get_factor("state_key").inputs_hr[traj_idx]
    else:
        door_open, key_pos, key_picked = None, None, None

    def viz(t):
        s_t_idx = state_traj[t]
        agent_dir = act_traj[t - 1] if t > 0 else 0
        door_open_t = door_open[t] if door_open is not None else None
        key_pos_t = key_pos[t] if key_pos is not None else None
        key_picked_t = key_picked[t] if key_picked is not None else None

        agent_img = display_agent(
            env,
            agent_pos=env.IDX_TO_POS[s_t_idx.item()],
            agent_dir=agent_dir,
            door_open=door_open_t,
            key_pos=key_pos_t,
            key_picked=key_picked_t,
        )
        display(Image.fromarray(agent_img))
        if act_traj is not None:
            assert rew_traj is not None
            a_t_idx, r_t_idx, rtg_t_idx = act_traj[t], rew_traj[t], rtg_traj[t]
            r = "NONE" if r_t_idx.isnan() else r_idx_to_r(env, r_t_idx)
            rtg = "NONE" if rtg_t_idx.isnan() else rtg_idx_to_rtg(env, rtg_t_idx)
            print(f"Action chosen {AGENT_DIR_TO_STR[a_t_idx.item()]}, reward {r}, rtg {rtg}")

    widgets.interact(viz, t=widgets.IntSlider(min=0, max=len(state_traj) - 1, step=1, value=0))


def visualize_predictions(
    env,
    b,
    start_pos=None,
    key_pos=None,
    show_agent_key=True,
    traj_idx=0,
    show_non_pred=False,
    gt_probs_info=False,
    save_prefix=None,
):
    # TODO: make this not depend on batch (?). That would be nice because it would mean that we can visualize
    #  completions this way too (which don't have a single sub-batch for them).
    # Should be a way to only have "visualize input data" and have it do everything.
    traj_states_in = b.get_factor("state").inputs_hr

    #    if env.base_env.__class__ is CustomDoorKeyEnv6x6:
    #        traj_key_pos_in = sb.get_factor("state_key_pos").inputs_hr[traj_idx][:1]
    #        # traj_door_status_in = sb.get_factor("state_door").inputs_hr
    #    else:
    #        traj_key_pos_in = None

    #    if start_pos is not None:
    #        env.reset()
    #        env.set_agent_pos(start_pos)
    #
    #    if key_pos is not None:
    #        env.set_key_position(key_pos)

    if not show_agent_key:
        env.grid.set(*key_pos, None)
        env.grid.set(*start_pos, None)
        key_pos = None
        start_pos = None

    traj_actions_in = b.get_factor("action").inputs_hr

    traj_states_probs = b.get_factor("state").prediction_probs
    traj_actions_probs = b.get_factor("action").prediction_probs

    traj_states_mask = (1 - b.get_input_mask_for_factor("state")).to(int)
    traj_actions_mask = (1 - b.get_input_mask_for_factor("action")).to(int)

    color = np.array([0, 0, 255])

    _visualize_state_predictions(
        env,
        traj_states_in[traj_idx],
        traj_actions_in[traj_idx],
        traj_states_probs[traj_idx],
        traj_states_mask[traj_idx],
        traj_actions_probs[traj_idx],
        traj_actions_mask[traj_idx],
        color,
        show_non_pred=show_non_pred,
        gt_probs_info=gt_probs_info,
        save_prefix=save_prefix,
        agent_pos=start_pos,
        key_pos=key_pos,
    )


#     traj_acts_in = sb.get_factor("action").inputs_hr
#     traj_acts_probs = sb.get_factor("action").prediction_probs
#     traj_acts_mask = (1 - sb.get_input_mask_for_factor("action")).to(int)
#
#     _visualize_action_predictions(
#         env, traj_acts_in, traj_acts_probs, traj_acts_mask, traj_idx=traj_idx
#     )
#
#
# def _visualize_action_predictions(
#     env, traj_acts_in, traj_acts_probs, traj_acts_mask, traj_idx=0
# ):
#     """
#     Currently not doing this, although might make sense to add at some point building off the code from
#     # here: https://inst.eecs.berkeley.edu/~cs188/su20/project3/
#     """
#     traj_acts_in = traj_acts_in[traj_idx]
#     traj_acts_probs = traj_acts_probs[traj_idx]
#     traj_acts_mask = traj_acts_mask[traj_idx]


def get_gt_prob_completion(env, horizon, initial_states=()):
    p_by_time = defaultdict(np.array)

    if len(initial_states) == 0:
        p_by_time[0] = np.array([1 / (env.NUM_STATES - 1)] * (env.NUM_STATES - 1) + [0])
    else:
        for t, state_idx in enumerate(initial_states):
            p_by_time[t] = np.eye(env.NUM_STATES)[state_idx]

    for t in range(len(initial_states), horizon):
        p_by_time[t] = np.zeros(env.NUM_STATES)

        for start_state_idx in range(env.NUM_STATES):

            for act_i, act_p in enumerate(StochGoalAgent(env, 1).get_action_probs(start_state_idx, env.GOAL_LOC)):
                next_state_idx = env.get_next_state(start_state_idx, act_i)

                # Add mass to the next timestep state in proportion to the probability of being at the initial
                # state and transitioning
                p_by_time[t][next_state_idx] += act_p * p_by_time[t - 1][start_state_idx]

        assert np.allclose(sum(p_by_time[t]), 1), p_by_time[t]

    visualize_state_distribs(env, p_by_time)


def viz_categorical_over_grid(
    env,
    distr,
    gt_s_t_idx=0,
    color=(0, 0, 255),
    show_distr=True,
    rescale_probs=True,
    agent_pos=None,
    key_pos=None,
    door_open=True,
    key_picked=False,
):
    assert isinstance(distr, Categorical)
    agent_img = display_agent(
        env,
        agent_pos=agent_pos,
        door_open=door_open,
        key_pos=key_pos,
        key_picked=key_picked,
    )

    if show_distr:
        max_prob = distr.probs.max().item()
        print(distr)
        for (x, y), p in distr.distr.items():
            alpha = p / max_prob if rescale_probs else p
            agent_img = color_xy_patch(agent_img, x, y, np.array(color), alpha)

    display(Image.fromarray(agent_img))
    return agent_img


def visualize_state_distribs(env, state_distrbs):
    """
    show_non_pred: show state predictions even for timesteps which were not ones used
    gt_probs_info: show also the ground truth probabilities for the agent's actions (used for generating the data)
    """

    def viz(t):
        curr_s_p = Categorical.from_domain_and_probs(env.VALID_POSITIONS, state_distrbs[t])
        viz_categorical_over_grid(env, distr=curr_s_p)

    widgets.interact(viz, t=widgets.IntSlider(min=0, max=len(state_distrbs) - 1, step=1, value=0))


def _visualize_state_predictions(
    env,
    gt_states,
    gt_actions,
    state_probs,
    state_masks,
    action_probs,
    action_masks,
    color,
    rescale_probs=True,
    show_non_pred=False,
    gt_probs_info=False,
    agent_pos=None,
    key_pos=None,
    save_prefix=None,
):
    """
    show_non_pred: show state predictions even for timesteps which were not ones used
    gt_probs_info: show also the ground truth probabilities for the agent's actions (used for generating the data)
    """

    def viz(t, return_img=False):
        gt_s_t_idx = gt_states[t].item()
        gt_a_t_idx = gt_actions[t].item()
        curr_s_p = Categorical.from_domain_and_probs(env.VALID_POSITIONS, state_probs[t])
        mask = state_masks[t]

        img = viz_categorical_over_grid(
            env,
            distr=curr_s_p,
            gt_s_t_idx=gt_s_t_idx,
            color=color,
            show_distr=mask or show_non_pred,
            rescale_probs=rescale_probs,
            agent_pos=agent_pos,
            key_pos=key_pos,
        )

        if mask:
            print("Currently predicting state")

        # Display ground truth action probabilities from the actual state
        # the agent was in
        from uniMASK.envs.minigrid.agents import StochGoalAgent

        action_probs_gt = StochGoalAgent(env, 1).get_action_probs(gt_s_t_idx, env.GOAL_LOC)
        if action_masks[t]:
            print("Predictions", get_action_probs_string(action_probs[t], gt_a_t_idx))

        if gt_probs_info:
            print("GT probs", get_action_probs_string(action_probs_gt, gt_a_t_idx))

        if return_img:
            return img

    if save_prefix is not None:
        for t in range(len(gt_states)):
            img = viz(t, return_img=True)
            save_cropped(img, f"{save_prefix}_{t}.png")

    widgets.interact(viz, t=widgets.IntSlider(min=0, max=len(gt_states) - 1, step=1, value=0))
    return viz


def get_action_probs_string(probs, to_highlight):
    """Prints the probabilities nicely and highlights one of them"""
    s = ""
    for i, p in enumerate(probs):
        p = str(round(p, 5))
        if i == to_highlight:
            p = format_str_to_red(p)
        s += "{}: {}\t".format(AGENT_DIR_TO_STR[i], p)
    return s


def visualize_action_predictions(env, gt_states, gt_acts, act_probs, masks, color, rescale_probs=True):
    def viz(t):
        s_t_idx = gt_states[t]
        a_t_idx = gt_acts[t].item()
        curr_prob = act_probs[t]
        mask = masks[t]

        x, y = tuple(env.IDX_TO_POS[s_t_idx.item()])
        agent_img = display_agent(env, (x, y))

        max_prob = curr_prob.max().item()

        for i, p in enumerate(curr_prob):
            alpha = p.item() / max_prob if rescale_probs else p.item()
            agent_img = color_xyd_patch(agent_img, x, y, i, color, alpha)

        display(Image.fromarray(agent_img))

        s = ""

        if not mask:
            s += "SEEN: state {}, action {}, and reward\\n".format((x, y), AGENT_DIR_TO_STR[a_t_idx])
            s += "NOT "

        s += "Predicting: "
        for i, p in enumerate(curr_prob):
            s += "{}: {:.2f}, ".format(AGENT_DIR_TO_STR[i], p.item())

        print(s)

    widgets.interact(viz, t=widgets.IntSlider(min=0, max=len(gt_states) - 1, step=1, value=0))


def visualize_completions(env, sa_transition_counter, start_pos=(0, 0), key_pos=None, save_path=None):
    from IPython.core.display import display
    from PIL import Image

    # TODO: maybe make a copy?
    env.reset()
    env.set_agent_pos(start_pos)

    if isinstance(env.base_env, CustomDoorKeyEnv6x6):
        assert key_pos is not None, "Set key position"
        env.set_key_position(key_pos)

    img = env.render(mode="rgb_array", highlight=False)

    max_count = max([max([c for c in d.values()]) for d in sa_transition_counter.values()])

    for s, actions_d in sa_transition_counter.items():
        for a, count in actions_d.items():
            x, y = env.IDX_TO_POS[s]
            img = color_transition(img, x, y, a, [0, 0, 0], count / max_count)

    display(Image.fromarray(img, "RGB"))

    if save_path is not None:
        save_cropped(img, save_path)


def visualize_trajectory(
    env,
    traj,
    start_pos=(0, 0),
    key_pos=None,
    goal_pos=None,
    waypoints_pos=None,
    num_interp_pts=5,
    draw_line=True,
    save_path=None,
):
    """Draws a single trajectory in time as a spline.

    Args:
        traj (FullTokenSeq)
        num_interp_pts (int): number of points in between grid cells to interpolate the trajectory viz
        draw_line (bool): True to draw line, False to draw points
    """
    assert traj.get_factor("state").inputs_hr.shape[0] == 1

    states = traj.get_factor("state").inputs_hr[0].tolist()
    actions = traj.get_factor("action").inputs_hr[0].tolist()

    # Set up image
    env.reset()
    env.set_agent_pos(start_pos)
    if isinstance(env.base_env, CustomDoorKeyEnv6x6):
        assert key_pos is not None, "Set key position"
        env.set_key_position(key_pos)
    # Visualize a novel goal position (e.g. for goal-conditioning)
    if goal_pos is not None:
        env.put_obj(Goal(), *goal_pos)
        env.grid.set(4, 4, None)
    im = env.render(mode="rgb_array", highlight=False)

    # Convert traj into list of x, y points to draw
    traj_px_pos = []
    for t, (state, ac) in enumerate(zip(states, actions)):
        x, y = env.IDX_TO_POS[state]
        px_x, px_y = get_pixel_coordinates_for_pos_center(x, y)
        # check whether we're in the same pos as last timestep
        if t > 0 and state == states[t - 1]:
            # draw an additional point into the wall
            d_vec = DIR_TO_VEC[ac]
            px_x + 0.75 * TILE_PIXELS * d_vec[0]
            px_y + 0.75 * TILE_PIXELS * d_vec[1]
        #            traj_px_pos.append((wall_px_x, wall_px_y))
        if t > 0:
            last_px_x, last_px_y = traj_px_pos[-1]
            xs = np.linspace(last_px_x, px_x, num_interp_pts) + np.random.normal(size=(num_interp_pts,))
            ys = np.linspace(last_px_y, px_y, num_interp_pts) + np.random.normal(size=(num_interp_pts,))
            interp_pos = list(zip(xs, ys))
            traj_px_pos.extend(interp_pos)
        else:
            traj_px_pos.append((px_x, px_y))

    plt.imshow(im)

    # Draw waypoints, if any
    if waypoints_pos is not None:
        way_pos_px = [get_pixel_coordinates_for_pos_center(x, y) for x, y in waypoints_pos]
        way_xs, way_ys = zip(*way_pos_px)
        plt.scatter(way_ys, way_xs, s=300, color="blue", alpha=0.5)

    # Draw traj
    ys, xs = list(zip(*traj_px_pos))  # note reversed plt axes
    if draw_line:
        # Interpolate points with a spline
        tck, u = splprep([xs, ys], s=0, k=4)
        # array of [xs, ys] for spline
        new_points = splev(u, tck)
        n = len(new_points[0])
        # Interpolate color indexing
        cm = plt.cm.get_cmap("plasma", len(new_points[0]))
        np.linspace(0, 1, n) ** 2
        # Draw the line with time-varying color by chopping pts into segments
        #  each segment will be drawn with a diff color
        #  lower = smoother color change
        s = 5  # Segment length
        for i in range(0, n - s, s):
            plt.plot(
                new_points[0][i : i + s + 1],
                new_points[1][i : i + s + 1],
                color=cm(i),
                linewidth=4,
                alpha=0.8,
            )
    else:
        cm = plt.cm.get_cmap("plasma", len(xs))
        plt.scatter(xs, ys, c=range(len(xs)), cmap=cm, alpha=0.8, s=20)

    if save_path is not None:
        # Crop and make it look nice
        margin = TILE_PIXELS // 2
        width, height, _ = im.shape
        plt.xlim(margin, width - margin)
        plt.ylim(height - margin, margin)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved to ", save_path)
    else:
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cm), ticks=[0, 1])
        cbar.ax.set_yticklabels(["t=0", "t=9"])

    plt.xlabel("x")
    plt.show()


# LOW LEVEL VIZ UTILS


def save_cropped(im, save_path):
    # Crop and make it look nice
    margin = TILE_PIXELS // 2
    width, height, _ = im.shape
    plt.imshow(im)
    plt.xlim(margin, width - margin)
    plt.ylim(height - margin, margin)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved to ", save_path)


def get_pixel_coordinates_for_pos(x, y):
    y_start = TILE_PIXELS * y
    y_end = y_start + TILE_PIXELS
    x_start = TILE_PIXELS * x
    x_end = x_start + TILE_PIXELS
    return (y_start, y_end, x_start, x_end)


def get_xy_patch(im, x, y):
    y_start, y_end, x_start, x_end = get_pixel_coordinates_for_pos(x, y)
    curr_patch = im[y_start:y_end, x_start:x_end]
    return curr_patch, (y_start, y_end, x_start, x_end)


def get_pixel_coordinates_for_pos_center(x, y):
    """Gets the pixel coordinates for the center of an (x,y) pos."""
    ys, ye, xs, xe = get_pixel_coordinates_for_pos(x, y)
    return (ys + ye) / 2, (xs + xe) / 2


def color_xy_patch(im, x, y, color, alpha):
    patch, xys = get_xy_patch(im, x, y)
    im[xys[0] : xys[1], xys[2] : xys[3]] = (1 - alpha) * patch + alpha * color
    return im


def color_xyd_patch(im, x, y, d, color, alpha):
    patch, xys = get_xy_patch_direction(im, x, y, d)
    im[xys[0] : xys[1], xys[2] : xys[3]] = (1 - alpha) * patch + alpha * color
    return im


def get_xy_patch_direction(im, x, y, d):
    y_start, y_end, x_start, x_end = get_xy_patch(im, x, y)[1]

    d_vec = DIR_TO_VEC[d]
    if np.array_equal(d_vec, [0, -1]):
        y_end -= 3 * TILE_PIXELS // 4
        x_start += TILE_PIXELS // 4
        x_end -= TILE_PIXELS // 4
    elif np.array_equal(d_vec, [0, 1]):
        y_start += 3 * TILE_PIXELS // 4
        x_start += TILE_PIXELS // 4
        x_end -= TILE_PIXELS // 4
    elif np.array_equal(d_vec, [1, 0]):
        x_start += 3 * TILE_PIXELS // 4
        y_start += TILE_PIXELS // 4
        y_end -= TILE_PIXELS // 4
    elif np.array_equal(d_vec, [-1, 0]):
        x_end -= 3 * TILE_PIXELS // 4
        y_start += TILE_PIXELS // 4
        y_end -= TILE_PIXELS // 4

    curr_patch = im[y_start:y_end, x_start:x_end]
    return curr_patch, (y_start, y_end, x_start, x_end)


def color_transition(im, x, y, d, color, alpha):
    patch, xys = get_xyd_transition_patch(im, x, y, d)
    color = np.array(color)
    im[xys[0] : xys[1], xys[2] : xys[3]] = (1 - alpha) * patch + alpha * color
    return im


LINE_WIDTH = TILE_PIXELS // 10
SIDE_PADDING = (TILE_PIXELS - LINE_WIDTH) // 2


def get_xyd_transition_patch(im, x, y, d):
    y_start, y_end, x_start, x_end = get_xy_patch(im, x, y)[1]

    d_vec = DIR_TO_VEC[d]
    if np.array_equal(d_vec, [0, -1]):  # UP
        x_start += SIDE_PADDING
        x_end -= SIDE_PADDING
        y_start -= SIDE_PADDING
        y_end -= SIDE_PADDING + LINE_WIDTH

    elif np.array_equal(d_vec, [0, 1]):  # DOWN
        x_start += SIDE_PADDING
        x_end -= SIDE_PADDING
        y_start += SIDE_PADDING + LINE_WIDTH
        y_end += SIDE_PADDING

    elif np.array_equal(d_vec, [1, 0]):  # RIGHT
        x_start += SIDE_PADDING + LINE_WIDTH
        x_end += SIDE_PADDING
        y_start += SIDE_PADDING
        y_end -= SIDE_PADDING

    elif np.array_equal(d_vec, [-1, 0]):  # LEFT
        x_start -= SIDE_PADDING
        x_end -= SIDE_PADDING + LINE_WIDTH
        y_start += SIDE_PADDING
        y_end -= SIDE_PADDING

    curr_patch = im[y_start:y_end, x_start:x_end]
    return curr_patch, (y_start, y_end, x_start, x_end)
