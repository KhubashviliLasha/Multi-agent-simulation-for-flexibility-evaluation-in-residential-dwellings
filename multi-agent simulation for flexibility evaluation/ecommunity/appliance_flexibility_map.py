import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import pandas as pd
import os
import openpyxl
import re
from collections import deque
import time
import logging

logging.basicConfig(level=logging.INFO)

class ApplianceFlexibilityGUI:
    def __init__(self, house_ids, db_path, excel_path, save_folder):
        self.house_ids = house_ids
        self.db_path = db_path
        self.excel_path = excel_path
        self.save_folder = save_folder
        
        self.df = None
        self.flexibility_info = None
        self.user_defined_flexibility = None
        self.question_queue = deque()
        
        self.root = None
        self.appliance_listbox = None
        self.validate_button = None

    def run(self):
        self.fetch_data()
        self.read_flexibility_info()
        self.setup_gui()
        self.root.mainloop()

    def fetch_data(self):
        conn = sqlite3.connect(self.db_path)
        query = """SELECT HouseIDREF, Name FROM Appliance WHERE HouseIDREF IN ({})""".format(','.join(['?']*len(self.house_ids)))
        self.df = pd.read_sql_query(query, conn, params=self.house_ids)
        conn.close()

    def read_flexibility_info(self):
        try:
            wb = openpyxl.load_workbook(self.excel_path)
            sheet = wb.active
            self.flexibility_info = {}
            headers = [cell.value for cell in sheet[1]]
            for row in sheet.iter_rows(min_row=2, values_only=True):
                appliance_name = row[0].strip()
                main_name = self.extract_main_name(appliance_name)
                self.flexibility_info[main_name] = {headers[i]: row[i] for i in range(1, len(headers))}
            self.user_defined_flexibility = self.flexibility_info.copy()
        except Exception as e:
            messagebox.showerror("Error", f"Error reading appliance info from Excel file:\n{e}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Appliance Flexibility Confirmation")
        
        appliance_listbox_label = tk.Label(self.root, text="Processed Appliances:")
        appliance_listbox_label.pack()
        
        self.appliance_listbox = tk.Listbox(self.root, width=50, height=10)
        self.appliance_listbox.pack()
        
        start_questions_button = ttk.Button(self.root, text="Start Processing", command=self.start_questions)
        start_questions_button.pack()
        
        self.validate_button = ttk.Button(self.root, text="Save to CSV", command=self.save_to_csv, state=tk.DISABLED)
        self.validate_button.pack()
        
        self.center_window(self.root, 800, 550)

    def start_questions(self):
        try:
            logging.info("Start Processing button clicked")
            self.question_queue.clear()
            for house_id in self.house_ids:
                filtered_df = self.df[self.df['HouseIDREF'] == int(house_id)]
                for index, row in filtered_df.iterrows():
                    appliance_name = row['Name']
                    main_name = self.extract_main_name(appliance_name)
                    current_condition = self.user_defined_flexibility.get(main_name, {})
                    self.question_queue.append((appliance_name, house_id, current_condition))
            if self.question_queue:
                logging.info(f"Processing {len(self.question_queue)} appliances")
                messagebox.showinfo("Processing Started", f"Starting to process {len(self.question_queue)} appliances.")
                self.root.after(100, self.ask_next_question)  # Schedule ask_next_question to run after 100ms
            else:
                logging.warning("No appliances to process")
                messagebox.showinfo("No Appliances", "No appliances to process.")
        except Exception as e:
            logging.error(f"Error in start_questions: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        

    def ask_next_question(self):
        if not self.question_queue:
            self.validate_button.config(state=tk.NORMAL)
            messagebox.showinfo("Processing Complete", "All appliances have been processed.")
            return
        appliance_name, house_id, current_condition = self.question_queue.popleft()
        self.show_question(appliance_name, house_id, current_condition)

    def show_question(self, appliance_name, house_id, current_condition):
        question_window = tk.Toplevel(self.root)
        question_window.title("Flexibility Confirmation")
        question_window.minsize(600, 400)
        frame = ttk.Frame(question_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        question_text = tk.Text(frame, wrap=tk.WORD, font=("Arial", 11), height=15, width=60)
        question_text.pack(fill=tk.BOTH, expand=True)
        
        question_label_text = f"House {house_id} appliance: {appliance_name} is:\n"
        for key, value in current_condition.items():
            if key is not None:
                question_label_text += f"{key}: {value}\n"
        question_label_text += "\nDo you agree?"
        
        question_text.insert(tk.END, question_label_text)
        question_text.config(state=tk.DISABLED)
        
        button_frame = ttk.Frame(question_window)
        button_frame.pack(pady=10)
        
        def on_yes():
            self.appliance_listbox.insert(tk.END, f"{appliance_name} (House {house_id}): {current_condition}")
            self.user_defined_flexibility[self.extract_main_name(appliance_name)] = current_condition
            question_window.destroy()
            self.ask_next_question()
        
        def on_no():
            question_window.destroy()
            self.ask_flexibility_condition(appliance_name, house_id, current_condition)
        
        yes_button = ttk.Button(button_frame, text="Yes", command=on_yes)
        yes_button.pack(side=tk.LEFT, padx=5)
        
        no_button = ttk.Button(button_frame, text="No", command=on_no)
        no_button.pack(side=tk.LEFT, padx=5)
        
        if self.question_queue:
            next_appliance_name, next_house_id, _ = self.question_queue[0]
            if next_house_id != house_id:
                next_message = f"Next: Switching to user {next_house_id}..."
                next_label = ttk.Label(question_window, text=next_message, font=("Arial", 9, "italic"))
                next_label.pack(pady=5)
        
        question_window.update_idletasks()
        width = max(question_window.winfo_reqwidth(), 600)
        height = max(question_window.winfo_reqheight(), 400)
        self.center_window(question_window, width, height)
        pass

    def ask_flexibility_condition(self, appliance_name, house_id, current_condition):
        def set_flexibility_condition():
            new_condition = {column: entry_vars[column].get() for column in entry_vars}
            if 'PERIOD OF USE' in new_condition:
                from_time = period_from_var.get()
                to_time = period_to_var.get()
                if from_time != "any" and to_time != "any":
                    new_condition['PERIOD OF USE'] = f"{from_time} - {to_time}"
                else:
                    new_condition['PERIOD OF USE'] = "any"
            self.appliance_listbox.insert(tk.END, f"{appliance_name} (House {house_id}): Updated")
            main_name = self.extract_main_name(appliance_name)
            self.user_defined_flexibility[main_name] = new_condition
            flexibility_window.destroy()
            self.ask_next_question()

        flexibility_window = tk.Toplevel(self.root)
        flexibility_window.title("Set Appliance Information")
        self.center_window(flexibility_window, 500, 500)

        question_label = tk.Label(flexibility_window, text=f"Set information for {appliance_name}:")
        question_label.pack(pady=10)

        entry_frame = ttk.Frame(flexibility_window)
        entry_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        entry_vars = {}
        row = 0
        columns_to_show = list(self.user_defined_flexibility[list(self.user_defined_flexibility.keys())[0]].keys())[:-1]

        for column in columns_to_show:
            label = ttk.Label(entry_frame, text=f"{column}:")
            label.grid(row=row, column=0, sticky="e", padx=5, pady=2)

            if column == "Flexibility":
                entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                entry = ttk.Combobox(entry_frame, textvariable=entry_vars[column], values=["flexible", "not_flexible", "Uncategorized"])
                entry['validate'] = 'key'
                entry['validatecommand'] = (entry.register(self.valid_input_flexibility), '%P')
            elif column in ["typical power", "typical annual energy [kwh/yr]"]:
                entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                entry = ttk.Entry(entry_frame, textvariable=entry_vars[column], validate="key")
                entry['validatecommand'] = (entry.register(self.validate_numeric_input), '%P')
            elif column == "PERIOD OF USE":
                period_frame = ttk.Frame(entry_frame)
                period_frame.grid(row=row, column=1, sticky="w", padx=5, pady=2)

                time_options = ["any"] + [f"{h:02d}:00" for h in range(24)]
                period_from_var = tk.StringVar(value="any")
                period_from = ttk.Combobox(period_frame, textvariable=period_from_var, values=time_options, width=5)
                period_from.grid(row=0, column=0, padx=(0, 5))

                ttk.Label(period_frame, text="to").grid(row=0, column=1, padx=5)

                period_to_var = tk.StringVar(value="any")
                period_to = ttk.Combobox(period_frame, textvariable=period_to_var, values=time_options, width=5)
                period_to.grid(row=0, column=2, padx=(5, 0))

                current_period = current_condition.get(column, "")
                if current_period and current_period != "any":
                    try:
                        from_time, to_time = current_period.split(" - ")
                        period_from_var.set(from_time)
                        period_to_var.set(to_time)
                    except ValueError:
                        # If splitting fails, set both to "any"
                        period_from_var.set("any")
                        period_to_var.set("any")

                entry_vars[column] = tk.StringVar()
            elif column == "comment":
                entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                comment_frame = ttk.Frame(entry_frame)
                comment_frame.grid(row=row, column=1, sticky="w", padx=5, pady=2)

                comment_type = ttk.Combobox(comment_frame, values=["no_comment", "custom"], width=10)
                comment_type.grid(row=0, column=0, padx=(0, 5))
                comment_type.set("no_comment" if current_condition.get(column, "") == "no_comment" else "custom")

                comment_entry = ttk.Entry(comment_frame, textvariable=entry_vars[column], width=30)
                comment_entry.grid(row=0, column=1)

                def update_comment_state(*args):
                    if comment_type.get() == "no_comment":
                        comment_entry.delete(0, tk.END)
                        comment_entry.config(state="disabled")
                        entry_vars[column].set("no_comment")
                    else:
                        comment_entry.config(state="normal")
                        if entry_vars[column].get() == "no_comment":
                            entry_vars[column].set("")

                comment_type.bind("<<ComboboxSelected>>", update_comment_state)
                update_comment_state()
            else:
                entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                entry = ttk.Combobox(entry_frame, textvariable=entry_vars[column], values=["yes", "no"])
                entry['validate'] = 'key'
                entry['validatecommand'] = (entry.register(self.valid_input_yes_no), '%P')

            if column not in ["PERIOD OF USE", "comment"]:
                entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)

            row += 1

        button_frame = ttk.Frame(flexibility_window)
        button_frame.pack(pady=10)

        save_button = ttk.Button(button_frame, text="Save", command=set_flexibility_condition)
        save_button.pack(side=tk.LEFT, padx=5)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=flexibility_window.destroy)
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Show next house ID message if applicable
        if self.question_queue:
            next_appliance_name, next_house_id, _ = self.question_queue[0]
            if next_house_id != house_id:
                next_message = f"Next: Switching to house {next_house_id}..."
                next_label = ttk.Label(flexibility_window, text=next_message, font=("Arial", 9, "italic"))
                next_label.pack(pady=5)

        flexibility_window.update_idletasks()
        width = max(flexibility_window.winfo_reqwidth(), 500)
        height = max(flexibility_window.winfo_reqheight(), 500)
        self.center_window(flexibility_window, width, height)

#
    def save_to_csv(self):
        os.makedirs(self.save_folder, exist_ok=True)
        for house_id in self.house_ids:
            file_path = os.path.join(self.save_folder, f"{house_id}.csv")
            if os.path.exists(file_path):
                answer = messagebox.askyesno("File Exists", f"{file_path} already exists. Do you want to overwrite it?")
                if not answer:
                    continue
            filtered_df = self.df[self.df['HouseIDREF'] == int(house_id)].copy()
            for column in self.user_defined_flexibility[list(self.user_defined_flexibility.keys())[0]].keys():
                filtered_df[column] = filtered_df['Name'].apply(lambda x: self.user_defined_flexibility.get(self.extract_main_name(x), {}).get(column, ""))
            try:
                filtered_df.to_csv(file_path, index=False)
            except PermissionError as e:
                messagebox.showerror("Permission Error", f"Permission denied: {e}")
                return
        messagebox.showinfo("Success", f"Data saved for all selected HouseIDREFs.")
        pass
    @staticmethod
    def extract_main_name(appliance_name):
        match = re.search(r'^(.*?)(?=\d|\()', appliance_name)
        if match:
            return match.group(0).strip()
        return appliance_name.strip()

    @staticmethod
    def center_window(window, width, height):
        window.update_idletasks()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    @staticmethod
    def validate_numeric_input(value):
        return value.isdigit() or value == ""
   
    @staticmethod
    def valid_input_yes_no(value):
        return value.lower() in ['yes', 'no', '']
    
    @staticmethod
    def valid_input_flexibility(value):
        return value.lower() in ['flexible', 'not_flexible', 'uncategorized', '']
    
    @staticmethod
    def validate_time_format(value):
        pattern = re.compile(r'^(\d{2}:\d{2} - \d{2}:\d{2})(, \d{2}:\d{2} - \d{2}:\d{2})*$')
        return pattern.match(value) is not None or value == ""






# for e communities 

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import re
import sqlite3

class FlexibilityApp:
    def __init__(self, db_path, excel_path2, save_folder2):
        self.db_path = db_path
        self.excel_path2 = excel_path2
        self.save_folder2 = save_folder2
        self.root = None
        self.current_index = 0
        self.flexibility_info = {}
        self.current_appliance = None
        self.e_communities = []
        self.current_e_community = None

    def run(self):
        self.root = tk.Tk()
        self.root.title("Flexibility Conditions")
        self.root.geometry("1000x800")
        self.center_window(self.root)
        self.ask_e_community_count()
        self.root.mainloop()

    def center_window(self, window):
        window.update_idletasks()
        window_width = 800
        window_height = 800
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def ask_e_community_count(self):
        self.frame = ttk.Frame(self.root)
        self.frame.pack(pady=20)
        self.label = ttk.Label(self.frame, text="How many E-communities do we need to have?")
        self.label.pack(pady=10)
        self.e_community_count_var = tk.IntVar()
        self.entry = ttk.Entry(self.frame, textvariable=self.e_community_count_var)
        self.entry.pack(pady=10)
        self.next_button = ttk.Button(self.frame, text="Next", command=self.ask_e_community_names)
        self.next_button.pack(pady=10)

    def ask_e_community_names(self):
        count = self.e_community_count_var.get()
        if count <= 0:
            messagebox.showerror("Error", "Please enter a valid number of E-communities")
            return
        self.e_community_count = count
        self.frame.destroy()
        self.frame = ttk.Frame(self.root)
        self.frame.pack(pady=20)
        self.label = ttk.Label(self.frame, text="Please define the name of each E-community")
        self.label.pack(pady=10)
        self.e_community_name_vars = []
        for i in range(count):
            label = ttk.Label(self.frame, text=f"E-community {i+1} name:")
            label.pack(pady=2)
            name_var = tk.StringVar()
            entry = ttk.Entry(self.frame, textvariable=name_var)
            entry.pack(pady=2)
            self.e_community_name_vars.append(name_var)
        self.next_button = ttk.Button(self.frame, text="Next", command=self.save_e_community_names)
        self.next_button.pack(pady=10)

    def save_e_community_names(self):
        self.e_communities = [var.get() for var in self.e_community_name_vars if var.get()]
        if not self.e_communities:
            messagebox.showerror("Error", "Please enter names for all E-communities")
            return
        self.frame.destroy()
        self.load_flexibility_info()

    def load_flexibility_info(self):
        try:
            # Load data from SQLite database
            appliance_df = self.load_sqlite_data()
            # Load data from Excel file
            flexibility_df = pd.read_excel(self.excel_path2)
            flexibility_df = flexibility_df.rename(columns={'Appliance': 'Name'})

            # Merge the dataframes
            merged_df = pd.merge(appliance_df, flexibility_df, left_on='FilteredName', right_on='Name', how='left')
            merged_df['Flexibility'] = merged_df['Flexibility'].fillna('uncategorized')
            merged_df = merged_df.fillna("")

            self.flexibility_info = merged_df.set_index('FilteredName').T.to_dict()

            self.current_e_community = self.e_communities.pop(0)
            self.display_next_appliance()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the file: {e}")
            self.root.destroy()

    def load_sqlite_data(self):
        conn = sqlite3.connect(self.db_path)
        query = """
        WITH CleanedNames AS (
            SELECT
                TRIM(SUBSTR(Name, 1, INSTR(Name, '(') - 1)) AS MainName
            FROM
                Appliance
        ),
        FilteredNames AS (
            SELECT
                CASE
                    WHEN MainName LIKE 'Non halogen lamp%' THEN 'Non halogen lamp'
                    WHEN MainName LIKE 'Halogen lamp%' THEN 'Halogen lamp'
                    ELSE MainName
                END AS FilteredName
            FROM
                CleanedNames
        )
        SELECT
            FilteredName
        FROM
            FilteredNames
        GROUP BY
            FilteredName
        ORDER BY
            FilteredName;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def display_next_appliance(self):
        if self.current_index < len(self.flexibility_info):
            self.current_appliance = list(self.flexibility_info.keys())[self.current_index]
            info = self.flexibility_info[self.current_appliance]
            display_text = f"The appliance {self.current_appliance} has the following recommendations:\n\n"
            for key, value in info.items():
                display_text += f"{key}: {value}\n"
            display_text += "\nDo you agree?"
            self.frame = ttk.Frame(self.root)
            self.frame.pack(pady=20)
            self.label = ttk.Label(self.frame, text=display_text)
            self.label.pack(pady=10)
            self.yes_button = ttk.Button(self.frame, text="Yes", command=self.agree)
            self.yes_button.pack(side=tk.LEFT, padx=10)
            self.no_button = ttk.Button(self.frame, text="No", command=self.modify)
            self.no_button.pack(side=tk.RIGHT, padx=10)
        else:
            self.save_to_csv()
            if self.e_communities:
                self.current_e_community = self.e_communities.pop(0)
                self.current_index = 0
                self.display_next_appliance()
            else:
                messagebox.showinfo("Completed", "All appliances reviewed")
                self.root.destroy()

    def agree(self):
        self.current_index += 1
        self.frame.destroy()
        self.display_next_appliance()

    def modify(self):
        self.modify_window = tk.Toplevel(self.root)
        self.modify_window.title(f"Modify - {self.current_appliance}")
        self.modify_window.geometry("800x800")
        window_width = 800
        window_height = 800
        screen_width = self.modify_window.winfo_screenwidth()
        screen_height = self.modify_window.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.modify_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.entry_vars = {}
        row = 0
        current_condition = self.flexibility_info[self.current_appliance]
        columns_to_show = ['Flexibility', 'typical power', 'typical annual energy [kwh/yr]', 
                           'SIGNATURE', 'DELAYABLE', 'CANCELLABLE', 'INTERRUPTIBLE', 'ANTICIPATORY', 
                           'STARTABLE', 'PERIOD OF USE', 'comment']
        entry_frame = ttk.Frame(self.modify_window)
        entry_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        for column in columns_to_show:
            label = ttk.Label(entry_frame, text=f"{column}:", anchor='center', justify='center')
            label.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)
            if column == "Flexibility":
                self.entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))                                                                 
                vcmd = (self.modify_window.register(self.valid_input_flexibility), '%P')       # To block inputs 
                entry = ttk.Combobox(entry_frame, textvariable=self.entry_vars[column], values=["flexible", "not_flexible", "uncategorized"], validate='key', validatecommand=vcmd)                   # to block inputs 
                entry.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)

            elif column == "typical power":
                self.entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                vcmd = (self.modify_window.register(self.validate_numeric_input), '%P')
                entry = ttk.Entry(entry_frame, textvariable=self.entry_vars[column], validate='key', validatecommand=vcmd)
                entry.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)

            elif column == "typical annual energy [kwh/yr]":
                self.entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                vcmd = (self.modify_window.register(self.validate_numeric_input), '%P')
                entry = ttk.Entry(entry_frame, textvariable=self.entry_vars[column], validate='key', validatecommand=vcmd)
                entry.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)

            elif column == "PERIOD OF USE":
                period_frame = ttk.Frame(entry_frame)
                period_frame.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)
                period_label = ttk.Label(period_frame, text="From:")
                period_label.pack(side=tk.LEFT, padx=5, pady=2)
                self.period_from_var = tk.StringVar(value="any")
                vcmd = (self.modify_window.register(self.valid_input_time), '%P') # to block inputs 
                period_from = ttk.Combobox(period_frame, textvariable=self.period_from_var, 
                                           values=["any", "00:00", "01:00", "02:00", "03:00", "04:00", 
                                                   "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                                                   "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", 
                                                   "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", 
                                                   "23:00"],validate='key', validatecommand=vcmd)
                period_from.pack(side=tk.LEFT, padx=5, pady=2)
                period_label = ttk.Label(period_frame, text="To:")
                period_label.pack(side=tk.LEFT, padx=5, pady=2)
                self.period_to_var = tk.StringVar(value="any")
                vcmd = (self.modify_window.register(self.valid_input_time), '%P') # to block inputs 
                period_to = ttk.Combobox(period_frame, textvariable=self.period_to_var, 
                                         values=["any", "00:00", "01:00", "02:00", "03:00", "04:00", 
                                                 "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                                                 "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", 
                                                 "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", 
                                                 "23:00"], validate='key', validatecommand=vcmd)
                period_to.pack(side=tk.LEFT, padx=5, pady=2)

            elif column == "comment":
                comment_frame = ttk.Frame(entry_frame)
                comment_frame.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)
                vcmd=(self.modify_window.register(self.valid_input_comment), '%P')   # to block input
                self.comment_type = ttk.Combobox(comment_frame, values=["no_comment", "comment"], validate='key', validatecommand=vcmd)
                self.comment_type.pack(side=tk.LEFT)
                self.comment_type.set("no_comment")
                self.entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                self.comment_entry = ttk.Entry(comment_frame, textvariable=self.entry_vars[column], width=30, justify='center')
                self.comment_entry.pack(side=tk.LEFT)
                self.comment_type.bind("<<ComboboxSelected>>", self.update_comment_state)
                self.update_comment_state()
            else:
                self.entry_vars[column] = tk.StringVar(value=current_condition.get(column, ""))
                vcmd=(self.modify_window.register(self.valid_input_yes_no), '%P')
                entry = ttk.Combobox(entry_frame, textvariable=self.entry_vars[column], values=["yes", "no"], validate='key', validatecommand=vcmd)
                entry.pack(side=tk.TOP, anchor=tk.CENTER, padx=5, pady=2)
        save_button = ttk.Button(self.modify_window, text="Save", command=self.save_modifications)
        save_button.pack(side=tk.TOP, pady=10)

    def update_comment_state(self, event=None):
        if self.comment_type.get() == "no_comment":
            self.comment_entry.config(state="disabled")
        else:
            self.comment_entry.config(state="normal")

    def save_modifications(self):
        new_condition = {column: self.entry_vars[column].get() for column in self.entry_vars}
        if 'PERIOD OF USE' in new_condition:
            from_time = self.period_from_var.get()
            to_time = self.period_to_var.get()
            if from_time != "any" and to_time != "any":
                new_condition['PERIOD OF USE'] = f"{from_time} - {to_time}"
            else:
                new_condition['PERIOD OF USE'] = "any"
        if self.comment_type.get() == "no_comment":
            new_condition['comment'] = "no_comment"
        else:
            new_condition['comment'] = self.entry_vars['comment'].get()
        self.flexibility_info[self.current_appliance] = new_condition
        self.modify_window.destroy()
        self.agree()

    def save_to_csv(self):
        columns = ['Appliance', 'Flexibility', 'typical power', 'typical annual energy [kwh/yr]', 
                   'SIGNATURE', 'DELAYABLE', 'CANCELLABLE', 'INTERRUPTIBLE', 'ANTICIPATORY', 
                   'STARTABLE', 'PERIOD OF USE', 'comment']
        updated_df = pd.DataFrame.from_dict(self.flexibility_info, orient='index')
        updated_df.reset_index(inplace=True)
        updated_df.rename(columns={'index': 'Appliance'}, inplace=True)
        updated_df = updated_df[columns]
        file_path = os.path.join(self.save_folder2, f"{self.current_e_community}.csv")
        if os.path.exists(file_path):
            if not self.prompt_overwrite(file_path):
                self.save_to_csv()
                return
        updated_df.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Flexibility conditions have been saved to: {file_path}")

    def prompt_overwrite(self, file_path):
        result = messagebox.askquestion("File Exists", f"{file_path} already exists. Do you want to overwrite it?", icon='warning')
        if result == 'yes':
            return True
        else:
            new_file_path = self.get_new_file_path(file_path)
            self.current_e_community = new_file_path.split('/')[-1].split('.csv')[0]
            return False

    def get_new_file_path(self, old_file_path):
        new_file_path = old_file_path
        counter = 1
        while os.path.exists(new_file_path):
            new_file_path = old_file_path.replace('.csv', f'_{counter}.csv')
            counter += 1
        return new_file_path

    @staticmethod
    def validate_numeric_input(value):
        return value.isdigit() or value == ""
    
    @staticmethod
    def valid_input_time(value):
        return value.lower() in ["any", "00:00", "01:00", "02:00", "03:00", "04:00", 
                                "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                                "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", 
                                "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", 
                                "23:00"]

    @staticmethod
    def valid_input_flexibility(value):
        return value.lower() in ['flexible', 'not_flexible', 'uncategorized']

    @staticmethod
    def valid_input_yes_no(value):
        return value.lower() in ['yes', 'no']
    
    @staticmethod
    def valid_input_comment (value):
        return value.lower() in ["no_comment",'comment']
    
    @staticmethod
    def validate_time_format(value):
        pattern = re.compile(r'^(\d{2}:\d{2} - \d{2}:\d{2})(, \d{2}:\d{2} - \d{2}:\d{2})*$')
        return pattern.match(value) is not None or value == ""

    @staticmethod
    def read_flexibility_info(file_path):
        df = pd.read_excel(file_path)
        flexibility_info = df.set_index('Appliance').T.to_dict()
        return flexibility_info

# This part should be outside the class definition
if __name__ == "__main__":
    root = tk.Tk()
    app = FlexibilityApp(root)
    root.mainloop()


