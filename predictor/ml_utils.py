import pickle
import numpy as np
import pandas as pd
from django.conf import settings
import os


class NutritionPredictor:
    _instance = None
    DEBUG = True

    MIN_MULTIPLIER = 0.95   
    MAX_MULTIPLIER = 1.05   


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load_models(self):
        if self._loaded:
            return

        model_path = settings.ML_MODEL_PATH
        food_path = settings.FOOD_DATASET_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model file not found at: {model_path}\n"
                f"Run 'python ml_models/train_model.py' first."
            )

        with open(model_path, 'rb') as f:
            self.artifacts = pickle.load(f)

        if os.path.exists(food_path):
            self.food_df = pd.read_csv(food_path)
            self._normalize_food_data()
            self.has_fiber = 'fiber' in self.food_df.columns
            self._build_food_index()
            self._print_diagnostics()
        else:
            self.food_df = None
            self.has_fiber = False
            self._food_index = {}
            self._meal_index = {}
            print(f"WARNING: Food dataset not found at "
                  f"{food_path}. No diet plans.")

        self._loaded = True
        print("ML models loaded successfully.")

    def reload(self):
        self._loaded = False
        self.load_models()

    def _normalize_food_data(self):
        df = self.food_df
        df.columns = (
            df.columns.str.lower().str.strip()
            .str.replace(' ', '_').str.replace('-', '_')
        )

        for col in ['meal', 'diet']:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.strip()
                    .str.lower()
                    .str.replace('-', '_')
                    .str.replace(' ', '_')
                )

        meal_map = {
            'breakfast': 'breakfast', 'lunch': 'lunch',
            'dinner': 'dinner', 'snack': 'snacks',
            'snacks': 'snacks', 'evening_snack': 'snacks',
            'morning_snack': 'snacks',
            'mid_morning_snack': 'snacks',
            'afternoon_snack': 'snacks',
        }
        if 'meal' in df.columns:
            df['meal'] = df['meal'].map(meal_map).fillna(df['meal'])

        diet_map = {
            'vegetarian': 'vegetarian', 'veg': 'vegetarian',
            'non_vegetarian': 'non_vegetarian',
            'non_veg': 'non_vegetarian',
            'nonveg': 'non_vegetarian',
            'nonvegetarian': 'non_vegetarian',
        }
        if 'diet' in df.columns:
            df['diet'] = df['diet'].map(diet_map).fillna(df['diet'])

        if 'food' in df.columns:
            df['food'] = df['food'].astype(str).str.strip()

        for col in ['calories', 'protein', 'carbs', 'fat']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col], errors='coerce'
                ).fillna(0)

        if 'fiber' in df.columns:
            df['fiber'] = pd.to_numeric(
                df['fiber'], errors='coerce'
            ).fillna(0)

        df = df[df['calories'] > 0].reset_index(drop=True)
        df = (
            df.sort_values('protein', ascending=False)
            .drop_duplicates(
                subset=['food', 'meal', 'diet'], keep='first'
            )
            .reset_index(drop=True)
        )
        self.food_df = df

    def _build_food_index(self):
        self._food_index = {}
        self._meal_index = {}
        if self.food_df is None:
            return
        for (meal, diet), group in self.food_df.groupby(
            ['meal', 'diet']
        ):
            self._food_index[(meal, diet)] = (
                group.reset_index(drop=True)
            )
        for meal, group in self.food_df.groupby('meal'):
            self._meal_index[meal] = (
                group.reset_index(drop=True)
            )

    def _print_diagnostics(self):
        df = self.food_df
        print("\n===== FOOD DATASET DIAGNOSTICS =====")
        print(f"  Total items : {len(df)}")
        print(f"  Columns     : {list(df.columns)}")
        if 'meal' in df.columns:
            print(f"  Meals       : {sorted(df['meal'].unique())}")
        if 'diet' in df.columns:
            print(f"  Diets       : {sorted(df['diet'].unique())}")
        print(f"  Has fiber   : {self.has_fiber}")
        for key, group in self._food_index.items():
            meal, diet = key
            n = len(group)
            flag = " ⚠ TOO FEW" if n < 3 else ""
            print(f"    {meal:>12} + {diet:<16}: {n} items{flag}")
        print("====================================\n")

    def predict(self, age, gender, height_cm, weight_kg,
                activity_level, purpose, diet_pref='both'):

        self.load_models()
        art = self.artifacts

        assert gender in ('Male', 'Female')
        assert activity_level in (
            'sedentary', 'lightly_active',
            'moderately_active', 'very_active',
            'extra_active',
        )
        assert purpose in (
            'weight_loss', 'weight_gain',
            'muscle_gain', 'maintain',
        )
        assert diet_pref in (
            'vegetarian', 'non_vegetarian', 'both',
        )

        height_m = height_cm / 100.0
        bmi_val = round(weight_kg / (height_m ** 2), 2)

        sex_factor = 1 if gender == 'Male' else 0
        bf = round(
            1.20 * bmi_val + 0.23 * age
            - 10.8 * sex_factor - 5.4, 1
        )
        bf = max(4.0, min(bf, 55.0))
        lbm_val = round(weight_kg * (1 - bf / 100), 1)

        if bmi_val < 16:
            bcat = 'Severe Thinness'
        elif bmi_val < 17:
            bcat = 'Moderate Thinness'
        elif bmi_val < 18.5:
            bcat = 'Underweight'
        elif bmi_val < 25:
            bcat = 'Normal'
        elif bmi_val < 30:
            bcat = 'Overweight'
        elif bmi_val < 35:
            bcat = 'Obese I'
        elif bmi_val < 40:
            bcat = 'Obese II'
        else:
            bcat = 'Obese III'

        target_bmi_val = self._target_bmi(
            bmi_val, purpose, gender
        )
        target_weight = round(
            target_bmi_val * (height_m ** 2), 1
        )
        weight_change = round(target_weight - weight_kg, 1)

        inches = height_cm / 2.54
        if gender == 'Male':
            ideal_w = round(
                50.0 + 2.3 * max(0, inches - 60), 1
            )
        else:
            ideal_w = round(
                45.5 + 2.3 * max(0, inches - 60), 1
            )

        gender_enc = art['le_gender'].transform([gender])[0]
        activity_enc = art['le_activity'].transform(
            [activity_level]
        )[0]
        purpose_enc = art['le_purpose'].transform(
            [purpose]
        )[0]

        features = pd.DataFrame([{
            'age': age,
            'gender': gender_enc,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'activity_level': activity_enc,
            'purpose': purpose_enc,
            'bmi': bmi_val,
            'body_fat_pct': bf,
            'lean_body_mass_kg': lbm_val,
            'bmi_x_activity': bmi_val * activity_enc,
            'weight_x_height': weight_kg * height_cm,
            'age_x_bmi': age * bmi_val,
            'lbm_x_activity': lbm_val * activity_enc,
        }])

        cal_pred = max(1200, int(round(
            art['calorie_model'].predict(features)[0]
        )))
        prot_pred = max(50, int(round(
            art['protein_model'].predict(features)[0]
        )))
        carb_pred = max(80, int(round(
            art['carbs_model'].predict(features)[0]
        )))
        fat_pred = max(30, int(round(
            art['fat_model'].predict(features)[0]
        )))

        macro_cal = (
            prot_pred * 4 + carb_pred * 4 + fat_pred * 9
        )
        if macro_cal > 0:
            scale = cal_pred / macro_cal
            prot_pred = max(50, int(round(prot_pred * scale)))
            carb_pred = max(80, int(round(carb_pred * scale)))
            fat_pred = max(30, int(round(fat_pred * scale)))

        if gender == 'Male':
            bmr = round(
                10 * weight_kg + 6.25 * height_cm
                - 5 * age + 5
            )
        else:
            bmr = round(
                10 * weight_kg + 6.25 * height_cm
                - 5 * age - 161
            )

        act_mult = {
            'sedentary': 1.2, 'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725, 'extra_active': 1.9,
        }
        tdee = round(bmr * act_mult[activity_level])

        water_bonus = {
            'sedentary': 0, 'lightly_active': 250,
            'moderately_active': 500,
            'very_active': 750, 'extra_active': 1000,
        }
        water_ml = round(
            weight_kg * 35 + water_bonus[activity_level]
        )
        fiber_g = round(cal_pred / 1000 * 14)

        if purpose == 'maintain' or abs(weight_change) < 0.5:
            weeks = 0
        else:
            weekly_rate = {
                'weight_loss': 0.7,
                'weight_gain': 0.35,
                'muscle_gain': 0.25,
            }
            weeks = max(
                1,
                round(abs(weight_change)
                      / weekly_rate[purpose]),
            )

        verify_cal = (
            prot_pred * 4 + carb_pred * 4 + fat_pred * 9
        )

        meal_targets = self._compute_meal_targets(
            cal_pred, prot_pred, carb_pred,
            fat_pred, purpose,
        )

        if self.DEBUG:
            print(
                f"\n── Generating plan: purpose={purpose}, "
                f"cal={cal_pred}, P={prot_pred}, "
                f"C={carb_pred}, F={fat_pred} ──"
            )
            for m, t in meal_targets.items():
                print(
                    f"    {m}: cal={t['calories']}, "
                    f"P={t['protein']}, C={t['carbs']}, "
                    f"F={t['fat']}"
                )

        diet_plans = {}
        if self.food_df is not None:
            if diet_pref in ['vegetarian', 'both']:
                plan, totals = self._generate_plan(
                    meal_targets, 'vegetarian', purpose,
                )
                accuracy = (
                    round(
                        (1 - abs(totals['calories'] - cal_pred)
                         / cal_pred) * 100, 1
                    ) if cal_pred > 0 else 0.0
                )
                diet_plans['vegetarian'] = {
                    'plan': plan,
                    'totals': totals,
                    'accuracy': accuracy,
                }

            if diet_pref in ['non_vegetarian', 'both']:
                plan, totals = self._generate_plan(
                    meal_targets, 'non_vegetarian', purpose,
                )
                accuracy = (
                    round(
                        (1 - abs(totals['calories'] - cal_pred)
                         / cal_pred) * 100, 1
                    ) if cal_pred > 0 else 0.0
                )
                diet_plans['non_vegetarian'] = {
                    'plan': plan,
                    'totals': totals,
                    'accuracy': accuracy,
                }

        result = {
            'bmi': bmi_val,
            'bmi_category': bcat,
            'target_bmi': target_bmi_val,
            'target_weight': target_weight,
            'weight_change': weight_change,
            'weight_change_abs': abs(weight_change),
            'weight_direction': (
                'gain' if weight_change > 0
                else 'lose' if weight_change < 0
                else 'maintain'
            ),
            'ideal_weight': ideal_w,
            'body_fat_pct': bf,
            'lean_body_mass': lbm_val,
            'weeks_to_goal': weeks,
            'bmr': bmr,
            'tdee': tdee,
            'calories': cal_pred,
            'protein': prot_pred,
            'carbs': carb_pred,
            'fat': fat_pred,
            'water_ml': water_ml,
            'water_liters': round(water_ml / 1000, 1),
            'fiber': fiber_g,
            'protein_pct': (
                round(prot_pred * 4 / cal_pred * 100, 1)
                if cal_pred > 0 else 0.0
            ),
            'carbs_pct': (
                round(carb_pred * 4 / cal_pred * 100, 1)
                if cal_pred > 0 else 0.0
            ),
            'fat_pct': (
                round(fat_pred * 9 / cal_pred * 100, 1)
                if cal_pred > 0 else 0.0
            ),
            'macro_check_cal': verify_cal,
            'diet_plans': diet_plans,
        }
        return result

    def _compute_meal_targets(self, total_cal, total_prot,
                              total_carbs, total_fat, purpose):
        splits = self._get_meal_splits(purpose)
        targets = {}
        for meal, frac in splits.items():
            targets[meal] = {
                'calories': int(round(total_cal * frac)),
                'protein': int(round(total_prot * frac)),
                'carbs': int(round(total_carbs * frac)),
                'fat': int(round(total_fat * frac)),
            }
        return targets

    def _target_bmi(self, current_bmi, purpose, gender):
        healthy_mid = 23.0 if gender == 'Male' else 22.0
        cb = current_bmi
        if purpose == 'weight_loss':
            if cb >= 35:
                return round(cb * 0.88, 1)
            elif cb >= 30:
                return round(cb * 0.90, 1)
            elif cb >= 25:
                return round(max(healthy_mid, cb - 2.5), 1)
            else:
                return round(max(18.5, cb - 1.0), 1)
        elif purpose == 'weight_gain':
            if cb < 17:
                return round(cb + 3.5, 1)
            elif cb < 18.5:
                return round(cb + 2.5, 1)
            elif cb < 22:
                return round(cb + 2.0, 1)
            elif cb < 25:
                return round(min(26.5, cb + 1.5), 1)
            else:
                return round(cb + 1.0, 1)
        elif purpose == 'muscle_gain':
            if cb < 19:
                return round(cb + 3.0, 1)
            elif cb < 23:
                return round(cb + 2.5, 1)
            elif cb < 26:
                return round(cb + 1.5, 1)
            else:
                return round(cb + 0.8, 1)
        else:
            return round(cb, 1)

    def _get_meal_splits(self, purpose):
        if purpose == 'weight_loss':
            return {
                'breakfast': 0.30, 'lunch': 0.35,
                'snacks': 0.10, 'dinner': 0.25,
            }
        elif purpose == 'weight_gain':
            return {
                'breakfast': 0.25, 'lunch': 0.30,
                'snacks': 0.20, 'dinner': 0.25,
            }
        elif purpose == 'muscle_gain':
            return {
                'breakfast': 0.25, 'lunch': 0.30,
                'snacks': 0.20, 'dinner': 0.25,
            }
        else:
            return {
                'breakfast': 0.25, 'lunch': 0.35,
                'snacks': 0.15, 'dinner': 0.25,
            }

    def _get_candidates(self, meal, diet_type):
        key = (meal, diet_type)
        if key in self._food_index and len(
            self._food_index[key]
        ) > 0:
            if self.DEBUG:
                print(
                    f"    ✓ {meal}+{diet_type}: "
                    f"{len(self._food_index[key])} items"
                )
            return self._food_index[key], 'exact'

        if meal in self._meal_index and len(
            self._meal_index[meal]
        ) > 0:
            if self.DEBUG:
                print(
                    f"    ⚠ No {diet_type} for '{meal}' "
                    f"— all diets "
                    f"({len(self._meal_index[meal])} items)"
                )
            return self._meal_index[meal], 'any_diet'

        same_diet = self.food_df[
            self.food_df['diet'] == diet_type
        ]
        if len(same_diet) > 0:
            if self.DEBUG:
                print(
                    f"    ⚠ No '{meal}' — all "
                    f"{diet_type} ({len(same_diet)} items)"
                )
            return (
                same_diet.reset_index(drop=True),
                'any_meal',
            )

        if self.DEBUG:
            print(
                f"    ✗ Fallback entire dataset "
                f"({len(self.food_df)} items)"
            )
        return self.food_df, 'all'

    #  SCORING — RAW CALORIE DISTANCE
    def _score_food_match(self, opts, target):

        t_cal = max(target['calories'], 1)
        t_prot = max(target['protein'], 1)
        t_carb = max(target['carbs'], 1)
        t_fat = max(target['fat'], 1)

        food_cal = opts['calories'].clip(lower=1)
        food_prot = opts['protein'].fillna(0)
        food_carb = opts['carbs'].fillna(0)
        food_fat = opts['fat'].fillna(0)

        # Percentage error on RAW values
        err_cal = ((food_cal - t_cal).abs() / t_cal)
        err_prot = ((food_prot - t_prot).abs() / t_prot)
        err_carb = ((food_carb - t_carb).abs() / t_carb)
        err_fat = ((food_fat - t_fat).abs() / t_fat)

        # Weighted distance
        # Calories most important, then protein,
        # carbs, fat
        distance = (
            err_cal * 4.0
            + err_prot * 3.0
            + err_carb * 2.0
            + err_fat * 1.5
        )

        return distance

    #  FIND COMPLEMENT FOOD
    def _find_complement(self, opts, gap, used_foods):
        if gap['calories'] < 50:
            return None

        available = opts[
            ~opts['food'].str.lower().isin(
                {f.lower() for f in used_foods}
            )
        ]
        if len(available) < 1:
            available = opts

        available = available.copy()
        available['gap_score'] = self._score_food_match(
            available, gap
        )

        max_cal = gap['calories'] * 1.3
        reasonable = available[
            available['calories'] <= max_cal
        ]

        if len(reasonable) == 0:
            reasonable = available.nsmallest(
                3, 'calories'
            )

        best_idx = reasonable['gap_score'].idxmin()
        return reasonable.loc[best_idx]

    def _generate_plan(self, meal_targets, diet_type,
                       purpose):
        plan = {}
        total_actual = {
            'calories': 0, 'protein': 0,
            'carbs': 0, 'fat': 0,
        }
        used_foods = set()

        remaining = {
            'calories': sum(
                t['calories'] for t in meal_targets.values()
            ),
            'protein': sum(
                t['protein'] for t in meal_targets.values()
            ),
            'carbs': sum(
                t['carbs'] for t in meal_targets.values()
            ),
            'fat': sum(
                t['fat'] for t in meal_targets.values()
            ),
        }

        meal_list = list(meal_targets.items())

        for i, (meal, target) in enumerate(meal_list):
            is_last = (i == len(meal_list) - 1)

            if is_last:
                effective_target = {
                    'calories': max(100, remaining['calories']),
                    'protein': max(5, remaining['protein']),
                    'carbs': max(10, remaining['carbs']),
                    'fat': max(3, remaining['fat']),
                }
            else:
                effective_target = target.copy()

            if self.DEBUG:
                print(
                    f"\n  {meal}: target "
                    f"cal={effective_target['calories']}, "
                    f"P={effective_target['protein']}, "
                    f"C={effective_target['carbs']}, "
                    f"F={effective_target['fat']}"
                )

            opts, match_type = self._get_candidates(
                meal, diet_type
            )
            if len(opts) == 0:
                if self.DEBUG:
                    print(f"    ✗ No candidates — skipping")
                continue

            opts = opts.copy()
            if used_foods:
                available = opts[
                    ~opts['food'].str.lower().isin(
                        {f.lower() for f in used_foods}
                    )
                ]
                if len(available) >= 2:
                    opts = available

            # ── Score by RAW macro distance ──
            opts = opts.copy()
            opts['distance'] = self._score_food_match(
                opts, effective_target
            )
            opts_sorted = opts.sort_values(
                'distance'
            ).reset_index(drop=True)

            # ── Pick primary food ──
            primary = opts_sorted.iloc[0]

            # ── Compute tight multiplier ──
            if primary['calories'] > 0:
                raw_mult = (
                    effective_target['calories']
                    / primary['calories']
                )
            else:
                raw_mult = 1.0

            multiplier = float(np.clip(
                raw_mult,
                self.MIN_MULTIPLIER,
                self.MAX_MULTIPLIER,
            ))

            p_cal = int(round(
                primary['calories'] * multiplier
            ))
            p_prot = int(round(
                primary['protein'] * multiplier
            ))
            p_carb = int(round(
                primary['carbs'] * multiplier
            ))
            p_fat = int(round(
                primary['fat'] * multiplier
            ))

            primary_name = (
                str(primary['food'])
                if pd.notna(primary['food'])
                else 'Unknown Item'
            )

            if self.DEBUG:
                top_n = min(5, len(opts_sorted))
                print(f"      Top {top_n} matches (raw):")
                for j in range(top_n):
                    r = opts_sorted.iloc[j]
                    tag = " ◀ PRIMARY" if j == 0 else ""
                    print(
                        f"        {j+1}. {r['food']:<30} "
                        f"[{int(r['calories'])}/"
                        f"{int(r['protein'])}/"
                        f"{int(r['carbs'])}/"
                        f"{int(r['fat'])}] "
                        f"dist={r['distance']:.3f}"
                        f"{tag}"
                    )

            # ── Check if we need a complement food ──
            gap = {
                'calories': effective_target['calories'] - p_cal,
                'protein': effective_target['protein'] - p_prot,
                'carbs': effective_target['carbs'] - p_carb,
                'fat': effective_target['fat'] - p_fat,
            }

            complement_info = None
            c_cal = c_prot = c_carb = c_fat = 0
            complement_name = None

            # If gap > 20% of target, find complement
            gap_pct = (
                gap['calories']
                / max(effective_target['calories'], 1)
            )

            if gap_pct > 0.20:
                complement = self._find_complement(
                    opts,
                    gap,
                    used_foods | {primary_name},
                )

                if complement is not None:
                    # Complement multiplier also tight
                    if complement['calories'] > 0:
                        c_mult = float(np.clip(
                            gap['calories']
                            / complement['calories'],
                            self.MIN_MULTIPLIER,
                            self.MAX_MULTIPLIER,
                        ))
                    else:
                        c_mult = 1.0

                    c_cal = int(round(
                        complement['calories'] * c_mult
                    ))
                    c_prot = int(round(
                        complement['protein'] * c_mult
                    ))
                    c_carb = int(round(
                        complement['carbs'] * c_mult
                    ))
                    c_fat = int(round(
                        complement['fat'] * c_mult
                    ))

                    complement_name = (
                        str(complement['food'])
                        if pd.notna(complement['food'])
                        else 'Side Item'
                    )

                    complement_info = {
                        'food': complement_name,
                        'calories': c_cal,
                        'protein': c_prot,
                        'carbs': c_carb,
                        'fat': c_fat,
                        'multiplier': round(c_mult, 2),
                    }

                    if self.DEBUG:
                        print(
                            f"      + Complement: "
                            f"{complement_name} "
                            f"[{c_cal}/{c_prot}/"
                            f"{c_carb}/{c_fat}] "
                            f"x{c_mult:.2f}"
                        )

            # ── Total for this meal ──
            meal_cal = p_cal + c_cal
            meal_prot = p_prot + c_prot
            meal_carb = p_carb + c_carb
            meal_fat = p_fat + c_fat

            # Serving note for primary
            if multiplier < 0.85:
                serving_note = "(reduced portion)"
            elif multiplier > 1.15:
                serving_note = "(larger portion)"
            else:
                serving_note = ""

            # ── Build food display name ──
            if complement_name:
                display_name = (
                    f"{primary_name} + {complement_name}"
                )
            else:
                display_name = primary_name

            # ── Match quality ──
            t = effective_target
            match_pct = round(
                (1 - (
                    abs(meal_cal - t['calories'])
                    / max(t['calories'], 1)
                    + abs(meal_prot - t['protein'])
                    / max(t['protein'], 1)
                    + abs(meal_carb - t['carbs'])
                    / max(t['carbs'], 1)
                    + abs(meal_fat - t['fat'])
                    / max(t['fat'], 1)
                ) / 4) * 100, 1
            )

            plan[meal] = {
                'food': display_name,
                'primary_food': primary_name,
                'calories': meal_cal,
                'protein': meal_prot,
                'carbs': meal_carb,
                'fat': meal_fat,
                'target_calories': t['calories'],
                'target_protein': t['protein'],
                'target_carbs': t['carbs'],
                'target_fat': t['fat'],
                'multiplier': round(multiplier, 2),
                'note': serving_note,
                'match_type': match_type,
                'match_pct': match_pct,
            }

            if complement_info:
                plan[meal]['complement'] = complement_info

            total_actual['calories'] += meal_cal
            total_actual['protein'] += meal_prot
            total_actual['carbs'] += meal_carb
            total_actual['fat'] += meal_fat

            used_foods.add(primary_name)
            if complement_name:
                used_foods.add(complement_name)

            remaining['calories'] -= meal_cal
            remaining['protein'] -= meal_prot
            remaining['carbs'] -= meal_carb
            remaining['fat'] -= meal_fat

        return plan, total_actual

    def _purpose_macro_ratios(self, purpose, total_cal):
        ratios = {
            'weight_loss': {
                'p': 0.35, 'c': 0.35, 'f': 0.30,
            },
            'weight_gain': {
                'p': 0.20, 'c': 0.50, 'f': 0.30,
            },
            'muscle_gain': {
                'p': 0.35, 'c': 0.40, 'f': 0.25,
            },
            'maintain': {
                'p': 0.25, 'c': 0.45, 'f': 0.30,
            },
        }
        r = ratios[purpose]
        return {
            'protein_g': int(total_cal * r['p'] / 4),
            'carbs_g': int(total_cal * r['c'] / 4),
            'fat_g': int(total_cal * r['f'] / 9),
        }


# Module-level singleton
predictor = NutritionPredictor()